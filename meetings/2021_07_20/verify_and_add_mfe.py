import argparse
import pandas as pd
from utils.rna_ss_utils import get_fe_struct, compute_fe, arr2db, one_idx2arr
from utils.util_global_struct import process_bb_old_to_new
from utils_s2_tree_search import BoundingBox
from utils.inference_s2 import stem_bbs2arr


def main(in_file, out_file):
    df = pd.read_pickle(in_file)

    data = []
    for _, row in df.iterrows():
        seq = row['seq']
        idxes, mfe, mfe_freq, ens_div = get_fe_struct(seq)


        df_target = process_bb_old_to_new(row['bounding_boxes'])
        df_stem = df_target[df_target['bb_type'] == 'stem']
        bbs = []
        for idx, r in df_stem.iterrows():
            bbs.append(BoundingBox(bb_x=r['bb_x'],
                                   bb_y=r['bb_y'],
                                   siz_x=r['siz_x'],
                                   siz_y=r['siz_y']))
        x_arr = stem_bbs2arr(bbs, len(seq))
        db_str, is_ambiguous = arr2db(x_arr)
        assert not is_ambiguous
        mfe_2 = compute_fe(seq, db_str)

        if mfe != mfe_2:
            print(seq)
            print(idxes)
            print(mfe)
            print(db_str)
            print(mfe_2)
            print('')

        row['mfe'] = mfe_2  # use mfe_2, higher precision
        data.append(row)

    data = pd.DataFrame(data)
    data.to_pickle(out_file, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        help='Path to training data file, should be in pkl.gz format')
    parser.add_argument('--out', type=str, help='Path to output file')

    args = parser.parse_args()

    main(args.data, args.out)



