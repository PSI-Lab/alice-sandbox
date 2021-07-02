import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import f1_score
from utils.rna_ss_utils import arr2db, one_idx2arr, compute_fe
from utils.inference_s2 import Predictor, process_row_bb_combo, stem_bbs2arr


def main(topk):
    df = pd.read_pickle('../2021_06_15/data/data_len60_test_1000_s1_stem_bb_combos.pkl.gz')

    # part 1, sanity check, examples where target structure is in top K
    print("part 1, sanity check, examples where target structure is in top K")
    for _, row in df.iterrows():
        seq, df_valid_combos, bb_combos, target_bbs, target_bb_inc, target_in_combo, target_in_topk = process_row_bb_combo(
            row, topk)

        # only check those with target_in_topk == True,
        # sanity check this result in f1=1?
        if target_in_topk:

            # extract prediction of target (won't error out since target_in_topk == True)
            idx_tgt = next(i for i, bb_combo in enumerate(bb_combos) if set(bb_combo) == set(target_bbs))

            # use RNAfold to compute FE
            db_str_tgt, _ = arr2db(stem_bbs2arr(bb_combos[idx_tgt], len(seq)))
            db_str_all = [arr2db(stem_bbs2arr(bb_combo, len(seq)))[0] for bb_combo in bb_combos]
            fe_tgt = compute_fe(seq, db_str_tgt)
            fe_all = [compute_fe(seq, db_str) for db_str in db_str_all]

            idx_best = np.nanargmin(fe_all)

            # sanity check for this particular case, FE should equal
            assert fe_tgt == fe_all[idx_best]
            # but structure might not!
            if idx_tgt != idx_best:
                print("FE equal but structure differ:")
                print("target:")
                pprint(target_bbs)
                print("predicted:")
                pprint(bb_combos[idx_best])

            target_bps = stem_bbs2arr(target_bbs, len(seq))
            pred_bps = stem_bbs2arr(bb_combos[idx_best], len(seq))
            idx = np.triu_indices(len(seq))
            f1s = f1_score(y_pred=pred_bps[idx], y_true=target_bps[idx])
            print(f"Target FE {fe_tgt}, best in topk FE {fe_all[idx_best]}, f1={f1s}")
            print("")

    # part 2, run on all examples
    print("part 2, run on all examples")
    f1s_all = []
    for row_idx, row in df.iterrows():
        if row_idx % 50 == 0:
            print(row_idx)

        seq, df_valid_combos, bb_combos, target_bbs, target_bb_inc, target_in_combo, target_in_topk = process_row_bb_combo(
            row, topk)

        # use RNAfold to compute FE
        db_str_tgt, _ = arr2db(stem_bbs2arr(target_bbs, len(seq)))
        db_str_all = [arr2db(stem_bbs2arr(bb_combo, len(seq)))[0] for bb_combo in bb_combos]
        # fe_tgt = compute_fe(seq, db_str_tgt)
        fe_all = [compute_fe(seq, db_str) for db_str in db_str_all]

        idx_best = np.nanargmin(fe_all)

        target_bps = stem_bbs2arr(target_bbs, len(seq))
        pred_bps = stem_bbs2arr(bb_combos[idx_best], len(seq))
        idx = np.triu_indices(len(seq))
        f1s = f1_score(y_pred=pred_bps[idx], y_true=target_bps[idx])
        f1s_all.append(f1s)
    f1s_all = pd.DataFrame({'f1s_all': f1s_all})
    print(f1s_all.describe().T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int,
                        help='top K')
    args = parser.parse_args()

    print(f"top K={args.topk}")
    main(args.topk)

