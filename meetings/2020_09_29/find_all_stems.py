"""
Find all (locally longest) stems
"""
import argparse
import numpy as np
import pandas as pd
from dgutils.pandas import add_column


def contiguous_regions(condition):
    # thanks to https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx


dna_mapping = str.maketrans("ACGT", "TGCA")
#
#
# def rev_comp(x):
#     return x.translate(dna_mapping)[::-1]


def base_compatible(a, b):
    # whether a-b can form base pair
    # canonical
    if a.translate(dna_mapping) == b:
        return True
    # G-T
    elif a == 'G' and b =='T':
        return True
    elif b == 'G' and a == 'T':
        return True
    else:
        return False


def all_stems(x):
    # clean up
    x = x.upper().replace('U', 'T')
    assert set(x).issubset(set(list('ACGT')))

    # # reverse complement
    # x_rc = rev_comp(x)

    # reverse
    x_rc = x[::-1]


    len_total = len(x)
    stems = []

    # shifting left
    for offset in range(len_total):
        x_shifted = x[:len_total-offset]
        # x_rc_shifted = x_rc[offset:]
        x_rc_shifted = x_rc[offset:]

        # array of 0/1 indicating whether the character is the same at the position
        idx = [1 if base_compatible(a, b) else 0 for a, b in zip(x_shifted, x_rc_shifted)]
        # find blocks of contiguous stretches of 1
        start_end = contiguous_regions(np.asarray(idx) == 1)

        # print(start_end)
        for se in start_end:
            s = se[0]
            e = se[1]
            # calculate stem location and length
            i = s
            j = len_total - offset - s
            l_stem = e- s
            stems.append({
                'i_start': i,
                'i_end': i + l_stem,  # open interval end
                'j_start': j - l_stem,
                'j_end': j,  # open interval end
                'l_stem': l_stem,
            })

    # shifting right
    for offset in range(len_total):
        x_shifted = x[offset:]
        x_rc_shifted = x_rc[:len_total-offset]

        # array of 0/1 indicating whether the character is the same at the position
        idx = [1 if base_compatible(a, b) else 0 for a, b in zip(x_shifted, x_rc_shifted)]
        # find blocks of contiguous stretches of 1
        start_end = contiguous_regions(np.asarray(idx) == 1)

        # print(start_end)
        for se in start_end:
            s = se[0]
            e = se[1]
            # calculate stem location and length
            i = s + offset
            j = len_total - s
            l_stem = e- s
            stems.append({
                'i_start': i,
                'i_end': i + l_stem,  # open interval end
                'j_start': j - l_stem,
                'j_end': j,  # open interval end
                'l_stem': l_stem,
            })

    # note that the above sliding window approach ensures that sub-optimal local alignment can only happen
    # at the boundary, e.g.
    #      GGGAAA
    # AAAGGG
    # in the above case, for the next sliding offset, the 2nd G on the top cannot align to the 2nd G in bottom ON ITSELF

    # above approach also ensures that if bounding box A is contained in B,
    # they need to share one side of the boundary
    # this makes merging much easier

    # use pandas for bb cleaning up
    df = pd.DataFrame(stems)

    # if no stems at all, just return empty list
    if len(df) == 0:
        return []

    # use consistent upper triangular representation
    # flip those ones if i > j
    # print('use consistent upper triangular representation')
    # swap i/j columns if i > j
    df.loc[df['i_start'] > df['j_start'], ['i_start', 'i_end', 'j_start', 'j_end']] = df.loc[
        df['i_start'] > df['j_start'], ['j_start', 'j_end', 'i_start', 'i_end']].values
    # remove duplicates
    df = df.drop_duplicates()
    # print(df)

    # merge: discard bounding box if it is fully included in another bounding box
    # print('merge')
    # sort first by start, then by end
    df = df.sort_values(by=['i_start', 'j_start', 'i_end', 'j_end'])
    # merge those with same starts, take last row (largest box)
    df = df.groupby(['i_start', 'j_start']).tail(1)
    # print(df)
    # sort first by end, then by start
    df = df.sort_values(by=['i_end', 'j_end', 'i_start', 'j_start'])
    # merge those with same ends, take first row (largest box)
    df = df.groupby(['i_end', 'j_end']).head(1)
    # print(df)

    stems_formatted = []
    for _, row in df.iterrows():
        i_start = row['i_start']
        j_start = row['j_start']
        i_end = row['i_end']
        j_end = row['j_end']
        l_stem = row['l_stem']
        stems_formatted.append({
            'bb_x': i_start,
            'bb_y': j_end - 1,   # location, not interval end!
            'siz_x': l_stem,
            'siz_y': l_stem,
        })

    return stems_formatted


def main(in_file, out_file):
    df = pd.read_pickle(in_file, compression='gzip')
    df = add_column(df, 'all_stems',
                    ['seq'], all_stems, pbar=True)

    df.to_pickle(out_file, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='input dataset with ground truth bounding boxes')
    parser.add_argument('--out_file', type=str, help='output dataset with all possible stem bounding boxes')
    args = parser.parse_args()
    main(args.in_file, args.out_file)







