# df with bb proposal -> df with valid global structs
import argparse
from util_global_struct import process_bb_old_to_new, add_bb_bottom_left, compatible_counts, LocalStructureBb, OneStepChain, GrowChain, GrowGlobalStruct, FullChain, chain_compatible, validate_global_struct
import numpy as np
import copy
from time import time
import pandas as pd
import dgutils.pandas as dgp
from util_global_struct import make_bb_df, generate_structs, filter_non_standard_stem


def check_bb_sensitivity(df_target, df_stem, df_iloop, df_hloop):
    # as a reference, check bounding box sensitivity
    n_found = 0
    for _, target_bb in df_target.iterrows():
        bb_x = target_bb['bb_x']
        bb_y = target_bb['bb_y']
        siz_x = target_bb['siz_x']
        siz_y = target_bb['siz_y']
        bb_type = target_bb['bb_type']
        if bb_type == 'stem':
            df_lookup = df_stem
        elif bb_type == 'iloop':
            df_lookup = df_iloop
        elif bb_type == 'hloop':
            df_lookup = df_hloop
        else:
            raise ValueError
        # try to find bb
        df_hit = df_lookup[(df_lookup['bb_x'] == bb_x) & (df_lookup['bb_y'] == bb_y) & (df_lookup['siz_x'] == siz_x) & (
                df_lookup['siz_y'] == siz_y)]
        if len(df_hit) == 1:
            n_found += 1
        elif len(df_hit) == 0:
            continue
        else:
            raise ValueError
    # print("Bounding box sensitivity: {} out of {}".format(n_found, len(df_target)))
    return n_found


def main(in_file, out_file, max_len, discard_ns_stem, min_hloop_size, min_pixel_pred, min_prob):
    # df = pd.read_pickle('../2020_09_22/data/rand_s1_bb_0p1.pkl.gz')
    df = pd.read_pickle(in_file)
    df_out = []

    # for debug
    err_idxes = []

    for idx, row in df.iterrows():
    # for idx, row in [(19301, df.iloc[19301])]:
    #     # debug
        if max_len !=0 and len(row['seq']) > max_len:  # default = 0 means no upper limit
            continue

        # print(idx, len(row['seq']))
        ctime = time()

        try:
            df_target = process_bb_old_to_new(row['bounding_boxes'])
            df_stem, df_iloop, df_hloop = make_bb_df(row['bb_stem'], row['bb_iloop'], row['bb_hloop'],
                                                     min_pixel_pred, min_prob)
            print("seq len {}, num bb's {}, {}, {}".format(len(row['seq']), len(df_stem), len(df_iloop), len(df_hloop)))

            # prune bounding boxes
            # stem - non standard base pairing
            if discard_ns_stem:
                n_before = len(df_stem)
                df_stem = filter_non_standard_stem(df_stem, row['seq'])
                print("df_stem base pair pruning, before: {}, after: {}".format(n_before, len(df_stem)))
            # hairpin loop - min size
            if min_hloop_size > 0:
                n_before = len(df_hloop)
                df_hloop = df_hloop[df_hloop['siz_x'] >= min_hloop_size]
                print("df_hloop min size pruning, before: {}, after: {}".format(n_before, len(df_hloop)))

            n_bb_found = check_bb_sensitivity(df_target, df_stem, df_iloop, df_hloop)
            # global_struct_dfs = generate_structs(df_stem, df_iloop, df_hloop)

            # # check whether ground truth is there
            # gt_found = False
            # for df_gs in global_struct_dfs:
            #     df_tmp = pd.merge(df_gs[['bb_x', 'bb_y', 'siz_x', 'siz_y', 'bb_type']], df_target, how='inner')
            #     if len(df_tmp) == len(df_target):
            #         # print("Ground truth!")
            #         gt_found = True

            row['df_target'] = df_target.to_dict('records')
            row['bb_stem'] = df_stem.to_dict('records')
            row['bb_iloop'] = df_iloop.to_dict('records')
            row['bb_hloop'] = df_hloop.to_dict('records')

            row['n_bb_found'] = n_bb_found
            # row['global_struct_dfs'] = [x.to_dict() for x in global_struct_dfs]  # convert to dicts for output
            # row['gt_found'] = gt_found
            df_out.append(row)
        except Exception as e:
            err_idxes.append((idx, str(e)))

        print("time: ", time() - ctime)

    df_out = pd.DataFrame(df_out)
    # print(df_out)
    df_out.to_pickle(out_file)

    print(err_idxes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input file, should be output from stage 1 (run_stage_1.py)')
    parser.add_argument('--max_len', type=int, default=0, help='sequence exceed max len will be skipped (for speed up testing)')
    parser.add_argument('--discard_ns_stem', action='store_true', help='Discard stems with non standard base pairs (anything other than A-U, G-C and G-U)')
    parser.add_argument('--min_hloop_size', type=int, default=1, help='Min hairpin loop size (note that size is always >=1)')
    parser.add_argument('--min_pixel_pred', type=int, default=3, help='pruning parameter')
    parser.add_argument('--min_prob', type=float, default=0.5, help='pruning parameter')
    parser.add_argument('--out_file', type=str, help='Path to output csv pickle')
    args = parser.parse_args()
    main(args.in_file, args.out_file, args.max_len, args.discard_ns_stem, args.min_hloop_size,  args.min_pixel_pred, args.min_prob)

