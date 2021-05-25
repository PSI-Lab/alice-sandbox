from collections import defaultdict
import argparse
import pandas as pd
from utils.rna_ss_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb
from utils.inference_s1_stem_bb import Predictor, Evaluator
from utils.util_global_struct import process_bb_old_to_new, filter_out_of_range_bb, filter_non_standard_stem, filter_diagonal_stem
import numpy as np
from utils.inference_s1_stem_bb import DataEncoder
from utils.misc import add_column, add_columns
# import dgutils.pandas as dgp
from collections import defaultdict


def check_stem_sensitivity_exact(df_target, df_stem):
    n_found = 0

    if len(df_stem) == 0:
        return n_found

    df_target = df_target[df_target['bb_type'] == 'stem']

    for _, target_bb in df_target.iterrows():
        bb_x = target_bb['bb_x']
        bb_y = target_bb['bb_y']
        siz_x = target_bb['siz_x']
        siz_y = target_bb['siz_y']

        # try to find bb
        df_hit = df_stem[(df_stem['bb_x'] == bb_x) & (df_stem['bb_y'] == bb_y) & (df_stem['siz_x'] == siz_x) & (
                df_stem['siz_y'] == siz_y)]
        if len(df_hit) == 1:
            n_found += 1
        elif len(df_hit) == 0:
            continue
        else:
            raise ValueError
    return n_found


def bb_boundary(bb_x, bb_y, siz_x, siz_y):
    x_s = bb_x
    y_e = bb_y + 1
    x_e = x_s + siz_x
    y_s = y_e - siz_y
    return x_s, x_e, y_s, y_e


def check_stem_sensitivity_within(df_target, df_stem):
    n_found = 0

    if len(df_stem) == 0:
        return n_found

    df_target = df_target[df_target['bb_type'] == 'stem']

    for _, target_bb in df_target.iterrows():
        x_s, x_e, y_s, y_e = bb_boundary(target_bb['bb_x'], target_bb['bb_y'],
                                         target_bb['siz_x'], target_bb['siz_y'])

        # add bb boundary, for easy compariso
        df_stem = add_columns(df_stem, ['x_s', 'x_e', 'y_s', 'y_e'],
                                  ['bb_x', 'bb_y', 'siz_x', 'siz_y'], bb_boundary)

        # try to find predicted bb that contains the target bb
        # note that it's not sufficient to have the bb1 contain bb2
        # their off-diagonal has to overlap
        df_hit = df_stem[(df_stem['x_s'] <= x_s) & (
                df_stem['x_e'] >= x_e) & (df_stem['y_s'] <= y_s) & (
                                 df_stem['y_e'] >= y_e) & (df_stem['x_e'] - x_e == y_s - df_stem['y_s'])]

        if len(df_hit) > 0:
            n_found += 1

    return n_found


def stem_bb_to_bp(bb_x, bb_y, siz_x, siz_y):
    # convert stem bb to base pair indices
    # follow convention that base pair (i, j) has i < j
    # to make life easier when looking for unique bps when merging multiple bbs
    bps = []
    x_s, x_e, y_s, y_e = bb_boundary(bb_x, bb_y, siz_x, siz_y)
    for ix, iy in zip(range(x_s, x_e), range(y_s, y_e)[::-1]):
        assert ix < iy
        bps.append((ix, iy))
    return bps


def main(in_file, out_file, model_path, threshold_on,
         compute_feature=False, include_all=False, num=None):
    # df = pd.read_pickle('../2021_03_23/data/debug_training_len20_200_100.pkl.gz')
    df = pd.read_pickle(in_file)

    # model_path = '../2021_03_23/s1_training/result/run_7/model_ckpt_ep_17.pth'  # best model

    # TODO load from input args?
    predictor = Predictor(model_ckpt=model_path,
                          num_filters=[128, 128, 256, 256, 512, 512],
                          filter_width=[3, 3, 5, 5, 7, 7],
                          hid_shared=[128, 128, 128, 256, 256, 256],
                          hid_output=[64], dropout=0)

    df_bps = []
    for idx, row in df.iterrows():
        # for debug use
        if num is not None and idx >= num:
            print("[debug] terminate parsing at idx {}".format(idx))
            break

        #     print(idx)
        seq = row['seq']
        one_idx = row['one_idx']
        bounding_boxes = row['bounding_boxes']
        df_target = process_bb_old_to_new(bounding_boxes)

        # threshold on p_on
        df_stem = predictor.predict_bb(seq=seq, threshold=threshold_on)

        # skip if no predicted bb
        if len(df_stem) == 0:
            print("Skip example with no predicted bb")
            continue

        # # check stem sensitivity
        # print("Idx {}, n_target_stem {}, n_exact_hit {}, n_within_hit {}".format(idx, len(
        #     df_target[df_target['bb_type'] == 'stem']),
        #                                                                          check_stem_sensitivity_exact(df_target,
        #                                                                                                       df_stem),
        #                                                                          check_stem_sensitivity_within(
        #                                                                              df_target, df_stem)))
        #
        # df_target_stem = df_target[df_target['bb_type'] == 'stem']
        #
        # if len(df_target[df_target['bb_type'] == 'stem']) > check_stem_sensitivity_within(df_target, df_stem):
        #     if not include_all:
        #         # only use examples with 100% sensitivy (for target equal/within pred bb)
        #         print("Skip example for now.")
        #         continue
        #     else:
        #         # use all, but in case sensitivity < 100%, subset target bb
        #         df_target_stem = pd.merge(df_target_stem, df_stem[['bb_x', 'bb_y', 'siz_x', 'siz_y']], how='inner')
        #         print("subset target stem bb to those in pred stem bb: {}".format(len(df_target_stem)))
        #
        # assert len(df_target_stem) == check_stem_sensitivity_within(df_target_stem, df_stem)
        # # assert len(df_target[df_target['bb_type'] == 'stem']) == check_stem_sensitivity_within(df_target, df_stem)

        if compute_feature:
            # extract base pair indices
            # pred
            # stem_bb_bps = []
            stem_bb_bps_prob = defaultdict(lambda: ([], [], [], []))  # 4 lists of prob
            # TODO add features (add ipput arg - only for one inference method)
            # TODO use dict
            # TODO summarize prob_on_sm         prob_other_sm             prob_on_sl          prob_other_sl
            for _, r in df_stem.iterrows():
                bps = stem_bb_to_bp(r['bb_x'], r['bb_y'], r['siz_x'], r['siz_y'])
                # stem_bb_bps.extend(bps)
                for bp in bps:
                    if 'prob_on_sm' in r:
                        stem_bb_bps_prob[bp][0].extend(r['prob_on_sm'])
                    if 'prob_other_sm' in r:
                        stem_bb_bps_prob[bp][1].extend(r['prob_other_sm'])
                    if 'prob_on_sl' in r:
                        stem_bb_bps_prob[bp][2].extend(r['prob_on_sl'])
                    if 'prob_other_sl' in r:
                        stem_bb_bps_prob[bp][3].extend(r['prob_other_sl'])
            # stem_bb_bps = sorted(list(set(stem_bb_bps)))  # remove duplicates (some pred bb might be within other pred bb)
            # bps
            stem_bb_bps = list(stem_bb_bps_prob.keys())
            # compute features
            stem_bb_bps_features = {}
            for bp in stem_bb_bps_prob:
                p1, p2, p3, p4 = stem_bb_bps_prob[bp]
                n1 = len(p1)
                n2 = len(p2)
                n3 = len(p3)
                n4 = len(p4)
                assert n1 == n2  # one is prob_on the other is other softmax probabilities
                assert n3 == n4
                p1_med = np.nanmedian(p1) if len(p1) > 0 else 0  # nanmedian still returns NaN if input list is empty
                p2_med = np.nanmedian(p2) if len(p2) > 0 else 0
                p3_med = np.nanmedian(p3) if len(p3) > 0 else 0
                p4_med = np.nanmedian(p4) if len(p4) > 0 else 0
                stem_bb_bps_features[bp] = (n1, n3, p1_med, p2_med, p3_med, p4_med)
        else:
            stem_bb_bps = []
            for _, r in df_stem.iterrows():
                bps = stem_bb_to_bp(r['bb_x'], r['bb_y'], r['siz_x'], r['siz_y'])
                stem_bb_bps.extend(bps)
            stem_bb_bps = sorted(list(set(stem_bb_bps)))
        # # target
        # target_bps = []
        # for _, r in df_target_stem.iterrows():
        #     bps = stem_bb_to_bp(r['bb_x'], r['bb_y'], r['siz_x'], r['siz_y'])
        #     target_bps.extend(bps)
        #
        # # double check that target_bps is subset of stem_bb_bps
        # assert set(target_bps).issubset(set(stem_bb_bps))

        # export
        row_new = row.copy()
        row_new['stem_bb_bps'] = stem_bb_bps
        # row_new['target_bps'] = target_bps
        if compute_feature:
            row_new['stem_bb_bps_features'] = stem_bb_bps_features
        df_bps.append(row_new)

    df_bps = pd.DataFrame(df_bps)
    df_bps.to_pickle(out_file, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--threshold_p', type=float,
                        help='threshold for stem on/off prob')
    # parser.add_argument('--threshold_n', type=float, default=-1,
    #                     help='threshold for stem n_proposal, default is to ignore this inference method')
    parser.add_argument('--model', type=str, help='Path to pytorch model params')
    parser.add_argument('--num', type=int, default=None, help='[debug use] max number of example to parse')
    parser.add_argument('--out_file', type=str, help='Path to output csv pickle')
    parser.add_argument('--features', action='store_true',
                        help='Specify this to compute s1 pred feature for each bp. For now only support inference method 1 using threshold_p')
    # parser.add_argument('--include_all', action='store_true',
    #                     help='Specify this to include all examples, regardless of their S1 bb sensitivity')
    args = parser.parse_args()
    # if args.threshold_p != -1:
    assert 0 < args.threshold_p < 1
    # if args.threshold_n != -1:
    #     assert 0 < args.threshold_n < 1
    # assert args.threshold_p != -1 or args.threshold_n != -1, "At least one of threshold_p and threshold_n need to be set!"
    # if args.features:
    #     assert args.threshold_n == -1
    main(args.data, args.out_file, args.model, threshold_on=args.threshold_p,
         compute_feature=args.features, include_all=args.include_all, num=args.num)
