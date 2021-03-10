import numpy as np
import pandas as pd


def merge_bbs(bbs):
    df = pd.DataFrame(bbs)
    df = df.groupby(['bb_x', 'bb_y', 'siz_x', 'siz_y'], as_index=False).agg(lambda x: [z for y in x for z in y])
    return df


def translate_bbs(bbs, ext_patch_row_start, ext_patch_col_start):
    result = []
    # in case nothing is being predicted
    if bbs is None:
        return result
    for bb in bbs:
        bb['bb_x'] += ext_patch_row_start
        bb['bb_y'] += ext_patch_col_start
        result.append(bb)
    return result


def predict_patch(seq, patch_row_start, patch_row_end, patch_col_start, patch_col_end, ext_patch_row_start, ext_patch_row_end, ext_patch_col_start, ext_patch_col_end, predictor_s1, threshold, topk, perc_cutoff):
    # # extract 'left' and 'right' sequence
    seq_1 = seq[ext_patch_row_start:ext_patch_row_end]
    seq_2 = seq[ext_patch_col_start:ext_patch_col_end]

    # index for picking output
    output_row_start = patch_row_start - ext_patch_row_start
    output_row_end = output_row_start + (patch_row_end - patch_row_start)
    output_col_start = patch_col_start - ext_patch_col_start
    output_col_end = output_col_start + (patch_col_end - patch_col_start)

    # make mask
    mask = np.zeros((ext_patch_row_end - ext_patch_row_start, ext_patch_col_end - ext_patch_col_start))
    # be careful if using negative index, to account for cases where ext_end = end
    # index for the region with 1's
    mask_row_start = patch_row_start - ext_patch_row_start
    mask_row_end = mask_row_start + (patch_row_end - patch_row_start)
    mask_col_start = patch_col_start - ext_patch_col_start
    mask_col_end = mask_col_start + (patch_col_end - patch_col_start)
    mask[mask_row_start:mask_row_end, mask_col_start:mask_col_end] = 1
    # print("Mask 1-region: {}-{}, {}-{}".format(mask_row_start, mask_row_end, mask_col_start, mask_col_end))

    ext_patch_stem, ext_patch_iloop, ext_patch_hloop = predictor_s1.predict_bb(seq_1, threshold, topk,
                                                                               perc_cutoff,
                                                                               seq2=seq_2, mask=mask)
    # translation
    patch_stem = translate_bbs(ext_patch_stem, ext_patch_row_start, ext_patch_col_start)
    patch_iloop = translate_bbs(ext_patch_iloop, ext_patch_row_start, ext_patch_col_start)
    patch_hloop = translate_bbs(ext_patch_hloop, ext_patch_row_start, ext_patch_col_start)
    return patch_stem, patch_iloop, patch_hloop


def predict_long_seq_wrapper(seq, patch_size, predictor_s1, threshold, topk=1, perc_cutoff=0, trim_size=None):
    # trim_size should be calculated automatically from model params
    if trim_size is not None:
        print("Warning: trim_size set by caller {}. Make sure you're debugging!".format(trim_size))
    else:
        trim_size = sum([x // 2 for x in predictor_s1.filter_width])

    n_splits = int(np.ceil(len(seq) / patch_size))
    # print(n_splits)  # total number of patches will be n_splits^2

    seq_len = len(seq)

    stems = []
    iloops = []
    hloops = []

    for idx_row in range(n_splits):
        for idx_col in range(n_splits):
            # top left corner
            # this is the output range we'll be extracting
            patch_row_start = idx_row * patch_size
            patch_col_start = idx_col * patch_size
            if patch_row_start + patch_size > seq_len:
                patch_row_end = seq_len
            else:
                patch_row_end = patch_row_start + patch_size
            if patch_col_start + patch_size > seq_len:
                patch_col_end = seq_len
            else:
                patch_col_end = patch_col_start + patch_size
            # this is what we feed into the NN, with enough context for conv layers
            if patch_row_start - trim_size < 0:
                ext_patch_row_start = 0
            else:
                ext_patch_row_start = patch_row_start - trim_size
            if patch_col_start - trim_size < 0:
                ext_patch_col_start = 0
            else:
                ext_patch_col_start = patch_col_start - trim_size
            # size (make sure to not go beyond the whole seq)
            if patch_row_start + patch_size + trim_size > seq_len:
                ext_patch_row_end = seq_len
            else:
                ext_patch_row_end = patch_row_start + patch_size + trim_size
            if patch_col_start + patch_size + trim_size > seq_len:
                ext_patch_col_end = seq_len
            else:
                ext_patch_col_end = patch_col_start + patch_size + trim_size

            # debug FIXME
            print("Input region: {}-{}, {}-{}".format(ext_patch_row_start, ext_patch_row_end, ext_patch_col_start,
                                                      ext_patch_col_end))
            print("Output region: {}-{}, {}-{}".format(patch_row_start, patch_row_end, patch_col_start,
                                                       patch_col_end))

            # get prediction for the patch
            patch_stem, patch_iloop, patch_hloop = predict_patch(seq, patch_row_start, patch_row_end, patch_col_start,
                                                                 patch_col_end, ext_patch_row_start, ext_patch_row_end,
                                                                 ext_patch_col_start, ext_patch_col_end, predictor_s1,
                                                                 threshold, topk, perc_cutoff)

            stems.extend(patch_stem)
            iloops.extend(patch_iloop)
            hloops.extend(patch_hloop)

    # make df, merge unique bb (since same bb could be predicted by adjacent patches)
    stems = merge_bbs(stems)
    iloops = merge_bbs(iloops)
    hloops = merge_bbs(hloops)

    return stems, iloops, hloops
