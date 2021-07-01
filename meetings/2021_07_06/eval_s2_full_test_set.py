import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from utils.rna_ss_utils import arr2db, one_idx2arr, compute_fe
from utils.inference_s2 import Predictor, process_row_bb_combo, stem_bbs2arr
from scipy.stats import pearsonr, spearmanr


def get_model(run_id):
    if run_id == 'run_11':
        model_path = '../2021_06_22/result/run_11/model_ckpt_ep_49.pth'
        model = Predictor(model_path,
                          num_filters=[16, 16, 32, 32, 64],
                          filter_width=[3, 3, 3, 3, 3],
                          pooling_size=[1, 1, 2, 2, 2])
    elif run_id == 'run_12':
        model_path = '../2021_06_29/result/run_12/model_ckpt_ep_49.pth'
        model = Predictor(model_path,
                          num_filters=[16, 16, 32, 32, 64],
                          filter_width=[3, 3, 3, 3, 3],
                          pooling_size=[1, 1, 2, 2, 2])
    elif run_id == 'run_13':
        model_path = '../2021_06_29/result/run_13/model_ckpt_ep_49.pth'
        model = Predictor(model_path,
                          num_filters=[16, 16, 32, 32, 64],
                          filter_width=[3, 3, 3, 3, 3],
                          pooling_size=[1, 1, 2, 2, 2])
    elif run_id == 'run_14':
        model_path = '../2021_06_29/result/run_14/model_ckpt_ep_99.pth'
        model = Predictor(model_path,
                          num_filters=[32, 32, 64, 64, 128],
                             filter_width=[3, 3, 3, 3, 3],
                             pooling_size=[1, 1, 2, 2, 2])
    elif run_id == 'run_15':
        model_path = '../2021_06_29/result/run_15/model_ckpt_ep_89.pth'
        model = Predictor(model_path,
                          num_filters=[64, 64, 128, 128, 256],
                             filter_width=[3, 3, 3, 3, 3],
                             pooling_size=[1, 1, 2, 2, 2])
    else:
        raise ValueError
    return model


def eval_test_set(model):

    # hard-coded constants
    TOPK = 100


    df = pd.read_pickle('../2021_06_15/data/data_len60_test_1000_s1_stem_bb_combos.pkl.gz')

    # part 1: evaluate on subset where target is within TOPK
    print("part 1: evaluate on subset where target is within TOPK")
    f1s_all = []
    for _, row in df.iterrows():
        seq, df_valid_combos, bb_combos, target_bbs, target_bb_inc, target_in_combo, target_in_topk = process_row_bb_combo(
            row, TOPK)

        # for now only check those with target_in_topk == True
        if target_in_topk:
            # prediction
            yp = model.predict_bb_combos(seq, bb_combos)
            idx_best_score = yp.argmax()

            # extract prediction of target (won't error out since target_in_topk == True)
            idx_tgt = next(i for i, bb_combo in enumerate(bb_combos) if set(bb_combo) == set(target_bbs))
            yp_tgt = yp[idx_tgt]
            # rank order
            idx_sort = np.argsort(yp)[::-1]
            rank_tgt = next(i for i in range(len(idx_sort)) if idx_sort[i] == idx_tgt)

            # use RNAfold to compute FE
            db_str_tgt, _ = arr2db(stem_bbs2arr(bb_combos[idx_tgt], len(seq)))
            db_str_pred, _ = arr2db(stem_bbs2arr(bb_combos[idx_best_score], len(seq)))
            fe_tgt = compute_fe(seq, db_str_tgt)
            fe_pred = compute_fe(seq, db_str_pred)

            # print("target:")
            # pprint(target_bbs)
            # print(f'target prediction: {yp_tgt}, rank: {rank_tgt} out of {len(yp)}, FE (RNAeval): {fe_tgt}')
            # print("prediction (best score):")
            # pprint(bb_combos[idx_best_score])
            # print(f"Best score: {yp.max()}, FE (RNAeval): {fe_pred}")

            target_bps = stem_bbs2arr(target_bbs, len(seq))
            pred_bps = stem_bbs2arr(bb_combos[idx_best_score], len(seq))
            idx = np.triu_indices(len(seq))
            f1s = f1_score(y_pred=pred_bps[idx], y_true=target_bps[idx])

            # print(f"f1 score: {f1s}")
            # print('')

            f1s_all.append(f1s)
    f1s_all = pd.DataFrame({'f1_score': f1s_all})
    print(f1s_all.describe().T)

    # part 2: all examples
    print("part 2: all examples")
    result_best_pred = []
    result_best_3 = []
    result_best_5 = []
    result_best_10 = []
    for _, row in df.iterrows():
        seq, df_valid_combos, bb_combos, target_bbs, target_bb_inc, target_in_combo, target_in_topk = process_row_bb_combo(
            row, TOPK)

        # check everything
        yp = model.predict_bb_combos(seq, bb_combos)
        # idx_best_score = yp.argmax()
        idx_sort_by_score = np.argsort(-yp)  # ascending on -yp --> descending on yp
        idx_sort_top = idx_sort_by_score[:10]

        # print("target:")
        # pprint(target_bbs)
        # print("prediction (best score):")
        # pprint(bb_combos[idx_best_score])

        target_bps = stem_bbs2arr(target_bbs, len(seq))
        # pred_bps = stem_bbs2arr(bb_combos[idx_best_score], len(seq))
        preds_bps = [stem_bbs2arr(bb_combos[idx], len(seq)) for idx in idx_sort_top]
        idx = np.triu_indices(len(seq))
        f1ss = [f1_score(y_pred=pred_bps[idx], y_true=target_bps[idx]) for pred_bps in preds_bps]

        # print(f"f1 score: {f1s}")
        # print('')

        # use RNAfold to compute FE
        db_str_tgt, _ = arr2db(stem_bbs2arr(target_bbs, len(seq)))
        db_strs_pred = []
        for idx in idx_sort_top:
            db_str_pred, _ = arr2db(stem_bbs2arr(bb_combos[idx], len(seq)))
            db_strs_pred.append(db_str_pred)

        fe_tgt = compute_fe(seq, db_str_tgt)
        fe_preds = [compute_fe(seq, db_str_pred) for db_str_pred in db_strs_pred]

        result_best_pred.append({
            'f1': f1ss[0],
            'target_fe': fe_tgt,
            'pred_fe': fe_preds[0],
        })

        idx_best_3 = np.argmax(f1ss[:3])
        result_best_3.append({
            'f1': f1ss[idx_best_3],
            'target_fe': fe_tgt,
            'pred_fe': fe_preds[idx_best_3],
        })

        idx_best_5 = np.argmax(f1ss[:5])
        result_best_5.append({
            'f1': f1ss[idx_best_5],
            'target_fe': fe_tgt,
            'pred_fe': fe_preds[idx_best_5],
        })

        idx_best_10 = np.argmax(f1ss[:10])
        result_best_10.append({
            'f1': f1ss[idx_best_10],
            'target_fe': fe_tgt,
            'pred_fe': fe_preds[idx_best_10],
        })

    result_best_pred = pd.DataFrame(result_best_pred)
    result_best_3 = pd.DataFrame(result_best_3)
    result_best_5 = pd.DataFrame(result_best_5)
    result_best_10 = pd.DataFrame(result_best_10)

    print("best-1 prediction:")
    print(result_best_pred[['f1']].describe().T)
    print("best-3 prediction:")
    print(result_best_3[['f1']].describe().T)
    print("best-5 prediction:")
    print(result_best_5[['f1']].describe().T)
    print("best-10 prediction:")
    print(result_best_10[['f1']].describe().T)

    for fe_threshold in [-5, -10, -15, -20, -25]:
        print(f"Target FE<={fe_threshold}:")
        print("best-1 prediction:")
        print(result_best_pred[result_best_pred['target_fe']<=fe_threshold][['f1']].describe().T)
        print("best-3 prediction:")
        print(result_best_3[result_best_3['target_fe']<=fe_threshold][['f1']].describe().T)
        print("best-5 prediction:")
        print(result_best_5[result_best_5['target_fe']<=fe_threshold][['f1']].describe().T)
        print("best-10 prediction:")
        print(result_best_10[result_best_10['target_fe']<=fe_threshold][['f1']].describe().T)


    # part 3: correlation between predicted score and RNAfold FE
    print("part 3: correlation between predicted score and RNAfold FE")
    # sample a few examples
    # remove those with very high FE (pseudoknot)
    # check correlation between (negative) predicted scores and RNAeval FE
    df_corr = []
    for _, row in df.sample(n=100).iterrows():  # sample 100 examples
        seq, df_valid_combos, bb_combos, target_bbs, target_bb_inc, target_in_combo, target_in_topk = process_row_bb_combo(
            row, TOPK)

        # prediction
        yp = model.predict_bb_combos(seq, bb_combos)
        fes = []
        for bb_combo in bb_combos:
            db_str, _ = arr2db(stem_bbs2arr(bb_combo, len(seq)))
            fe = compute_fe(seq, db_str)
            fes.append(fe)

        df_plot = pd.DataFrame({
            'pred': yp,
            'fe': fes,
        })

        df_plot = df_plot[df_plot['fe'] < 1000]

        # correlation
        p_corr, p_pval = pearsonr(-df_plot['pred'], df_plot['fe'])
        s_corr, s_pval = spearmanr(-df_plot['pred'], df_plot['fe'])

        df_corr.append({
            'p_corr': p_corr,
            'p_pval': p_pval,
            's_corr': s_corr,
            's_pval': s_pval,
        })

    df_corr = pd.DataFrame(df_corr)
    print(df_corr.describe().T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str,
                        help='run_id')
    args = parser.parse_args()

    model = get_model(args.run_id)
    print(f"Evaluating model: {args.run_id}")
    eval_test_set(model)

