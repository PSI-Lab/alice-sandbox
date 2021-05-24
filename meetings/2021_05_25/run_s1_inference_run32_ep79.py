import re
import argparse
import pandas as pd
from utils.inference_s1_stem_bb import Predictor, Evaluator


def compute_metric_summary(df, threhsold):
    # TODO load from input args?
    predictor = Predictor(model_ckpt='../2021_05_18/s1_training/result/run_32/model_ckpt_ep_79.pth',
                          num_filters=[128, 128, 256, 256, 512, 512],
                          filter_width=[3, 3, 5, 5, 7, 7],
                          hid_shared=[128, 128, 128, 256, 256, 256],
                          hid_output=[64], dropout=0)

    evaluator = Evaluator(predictor=None)

    df_metric = []

    for _, row in df.iterrows():
        seq = row['seq']
        bounding_boxes = row['bounding_boxes']
        df_target_stem = evaluator.make_target_bb_df(bounding_boxes, convert_tl_to_tr=True)
        df_pred = predictor.predict_bb(seq=seq, threshold=threhsold, filter_valid=True)
        m = evaluator.calculate_bb_metrics(df_target_stem, df_pred[['bb_x', 'bb_y', 'siz_x', 'siz_y']])
        s_bp = evaluator.calculate_bp_metrics(df_target_stem, df_pred, len(seq))
        s1 = m['n_target_identical'] / m['n_target_total']
        s2 = (m['n_target_identical'] + m['n_target_local'] + m['n_target_overlap']) / m['n_target_total']
        #     print(s1, s2)
        df_metric.append({
            'bb_identical': s1,
            'bb_overlap': s2,
            'bp': s_bp,
            'num_bb_ratio': len(df_pred) / len(df_target_stem)
        })

    df_metric = pd.DataFrame(df_metric)

    metric_summary = pd.DataFrame([{
        'threshold': threhsold,
        'bb_identical_mean': df_metric['bb_identical'].mean(),
        'bb_identical_std': df_metric['bb_identical'].std(),
        'bb_overlap_mean': df_metric['bb_overlap'].mean(),
        'bb_overlap_std': df_metric['bb_overlap'].std(),
        'bp_mean': df_metric['bp'].mean(),
        'bp_std': df_metric['bp'].std(),

    }])

    return metric_summary


def main(in_file, threshold, out_file):
    df = pd.read_pickle(in_file)

    metric_summary = compute_metric_summary(df, threshold)
    print(metric_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='test dataset to evaluate')
    parser.add_argument('--threshold', type=float, help='s1 inference threshold for p_on')
    # parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()
    assert 0 <= args.threshold <= 1
    main(args.data, args.threshold, args.out)




