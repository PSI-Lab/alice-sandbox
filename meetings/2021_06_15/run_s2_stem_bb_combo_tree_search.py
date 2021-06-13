import argparse
import numpy as np
import pandas as pd
from utils_s2_tree_search import StemBbTree, BoundingBox, bb_conflict
from utils.util_global_struct import process_bb_old_to_new


def get_total_bp(bb_inc, bbs):
    # get size
    sizes = [bbs[x].siz_x for x in bb_inc]
    return sum(sizes)


def main(in_file, out_file):
    df = pd.read_pickle(in_file)

    data = []

    for example_idx, row in df.iterrows():

        df_stem = pd.DataFrame(row.pred_stem_bb)
        print(example_idx, len(df_stem))

        # we use df index, make sure it's contiguous
        assert df_stem.iloc[-1].name == len(df_stem) - 1

        bbs = {}
        for idx, r in df_stem.iterrows():
            bbs[idx] = BoundingBox(bb_x=r['bb_x'],
                                   bb_y=r['bb_y'],
                                   siz_x=r['siz_x'],
                                   siz_y=r['siz_y'])

        bb_conf_arr = np.zeros((len(bbs), len(bbs)))
        for i in bbs.keys():
            bb1 = bbs[i]
            for j in bbs.keys():
                bb2 = bbs[j]
                # TODO only need to compute half
                bb_conf_arr[i, j] = bb_conflict(bb1, bb2)
        assert np.all(bb_conf_arr.T == bb_conf_arr)

        bb_tree = StemBbTree(bbs, bb_conf_arr)

        # find the target bb combo (might not be in the list of valid combo of pred bbs)
        df_target = process_bb_old_to_new(row.bounding_boxes)
        df_target = df_target[df_target['bb_type'] == 'stem']
        bb_idx_tgt = []
        tgt_in_pred_combo = True
        for _, r in df_target.iterrows():
            tgt_bb = BoundingBox(bb_x=r['bb_x'],
                                 bb_y=r['bb_y'],
                                 siz_x=r['siz_x'],
                                 siz_y=r['siz_y'])
            try:
                tgt_idx = next(i for i, bb in bbs.items() if bb == tgt_bb)
            except StopIteration:
                tgt_in_pred_combo = False
                bb_idx_tgt.append(tgt_idx)
        if tgt_in_pred_combo:
            bb_idx_tgt = set(bb_idx_tgt)
        else:
            bb_idx_tgt = None

        bb_combos = []
        for leaf in bb_tree.leaves:
            bb_inc = np.where(leaf.bb_assignment)[0].tolist()
            num_bps = get_total_bp(bb_inc, bbs)
            if tgt_in_pred_combo:
                is_target = set(bb_inc) == bb_idx_tgt
            else:
                is_target = False
            bb_combos.append({'bb_inc': bb_inc, 'num_bps': num_bps, 'is_target': is_target})

        row_new = row.copy()
        row_new['num_bb_combos'] = len(bb_combos)
        row_new['bb_combos'] = bb_combos
        row_new['tgt_in_pred_combo'] = tgt_in_pred_combo
        data.append(row_new)

    data = pd.DataFrame(data)
    data.to_pickle(out_file, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='test dataset to evaluate')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()
    main(args.data, args.out)


