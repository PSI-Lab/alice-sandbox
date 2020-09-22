"""
Prepare data for stage 2 model training
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def format_bb(bbs):
    # convert top left corner reference format (old) to top right reference format (new)
    # also return list of dict with keys: bb_x, bb_y, siz_x, siz_y, struct_type (can be easily converted to df)
    # bbs: list of e.g. ((9, 33), (2, 2), 'stem')
    result = []
    for (x0, y0), (wx, wy), struct_type in bbs:
        result.append({
            'bb_x': x0,
            'bb_y': y0 + wy - 1,
            'siz_x': wx,
            'siz_y': wy,
            'struct_type': struct_type,
        })
    return result


def compute_features(df, struct_type, start_idx=2):
    # first 3 index reserved for start and end
    data = []
    for _, row in df.iterrows():
        features = []
        if struct_type == 'stem':
            features.extend([0, 0, 0, 0, 1])
        elif struct_type == 'iloop':
            features.extend([0, 0, 0, 1, 0])
        elif struct_type == 'hloop':
            features.extend([0, 0, 1, 0, 0])
        else:
            raise ValueError  # first 2 bits reserved for start and end
        # 'closing' positions
        x = row['bb_x']
        y = row['bb_y']
        wx = row['siz_x']
        wy = row['siz_y']
        x0 = x + wx - 1
        y0 = y - wy + 1
        features.extend([x0, y0, x, y])
        # TODO 'closing' base pairs
        # TODO internal sequences?
        prob = row['prob']
        features.extend([np.max(prob), np.mean(prob), np.median(prob), np.min(prob)])  # std?
        new_row = row.copy()
        new_row['feature'] = features
        new_row['idx'] = start_idx
        data.append(new_row)
        start_idx += 1
    data = pd.DataFrame(data)
    return data


def convert_bb(x, y, wx, wy):
    # input: top right corner + size
    # return dict:
    # Keys: {'x1', 'x2', 'y1', 'y2'}
    # The (x1, y1) position is at the top left corner,
    # the (x2, y2) position is at the bottom right corner
    return {'x1': x,
            'x2': x + wx,  # open-right
            'y1': y - wy,  # open-right
            'y2': y}


#     return {'x1': x,
#             'x2': x + wx - 1,
#             'y1': y - wy + 1,
#             'y2': y}


def get_iou(bb1, bb2):
    """
    From https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    FIXME might contain bugs!

    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2'], bb1
    assert bb1['y1'] < bb1['y2'], bb1
    assert bb2['x1'] < bb2['x2'], bb2
    assert bb2['y1'] < bb2['y2'], bb2

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def find_closest_bb(df1, df2, return_col='idx'):
    # for each row in df1, find closest in df2, return data in return_col
    # FIXME note this is quite sub optimal, since if one bb shifts by 1 pixel, then the surrounding ones
    # has to shift in thoery to satisfy the hard constraints
    # indeed, we should not be looking for individual closest bb,
    # but the glbal cloest VALID path
    if len(df2) == 0:
        return [-1] * len(df1), [0] * len(df1)
    else:
        result_idx = []
        result_overlap = []
        for _, row in df1.iterrows():
            bb1 = convert_bb(row['bb_x'], row['bb_y'], row['siz_x'], row['siz_y'])
            overlaps = []
            for _, row2 in df2.iterrows():
                bb2 = convert_bb(row2['bb_x'], row2['bb_y'], row2['siz_x'], row2['siz_y'])
                overlaps.append((get_iou(bb1, bb2), bb2, row2[return_col]))
            # find largest overlap
            best_bb = max(overlaps, key=lambda x: x[0])
    #         print(bb1, best_bb)
            result_idx.append(best_bb[-1])
            result_overlap.append(best_bb[0])
        return result_idx, result_overlap


def process_row(row):
    bounding_boxes = pd.DataFrame(format_bb(row.bounding_boxes))
    if not isinstance(row.bb_stem, list) and pd.isnull(row.bb_stem):
        df_stem = pd.DataFrame([])
    else:
        df_stem = compute_features(pd.DataFrame(row.bb_stem), struct_type='stem', start_idx=2)

    if not isinstance(row.bb_iloop, list) and pd.isnull(row.bb_iloop):
        df_iloop = pd.DataFrame([])
    else:
        df_iloop = compute_features(pd.DataFrame(row.bb_iloop), struct_type='iloop', start_idx=2 + len(df_stem))

    if not isinstance(row.bb_hloop, list) and pd.isnull(row.bb_hloop):
        df_hloop = pd.DataFrame([])
    else:
        df_hloop = compute_features(pd.DataFrame(row.bb_hloop), struct_type='hloop',
                                start_idx=2 + len(df_stem) + len(df_iloop))
    
    # assert df_stem.iloc[-1].idx + 1 == df_iloop.iloc[0].idx
    # assert df_iloop.iloc[-1].idx + 1 == df_hloop.iloc[0].idx

    # finding index for each bounding box in ground truth
    target_idx = []
    bb_overlap = []
    idx, overlap = find_closest_bb(bounding_boxes[bounding_boxes['struct_type'] == 'stem'], df_stem)
    target_idx.extend(idx)
    bb_overlap.extend(overlap)
    idx, overlap = find_closest_bb(bounding_boxes[bounding_boxes['struct_type'].isin(['internal_loop', 'bulge'])],
                                   df_iloop)
    target_idx.extend(idx)
    bb_overlap.extend(overlap)
    idx, overlap = find_closest_bb(bounding_boxes[bounding_boxes['struct_type'] == 'hairpin_loop'], df_hloop)
    target_idx.extend(idx)
    bb_overlap.extend(overlap)

    # add in fixed start & end entry
    target_idx = [0] + target_idx + [1]
    bb_overlap = [1] + bb_overlap + [1]

    # features
    # the first 2 entries are place-holder for start & end, which have their special 'type' encoding
    # currently 13 features
    f0 = np.zeros((1, 13))
    f0[0, 0] = 1
    f1 = np.zeros((1, 13))
    f1[0, 1] = 1
    # features = np.concatenate([
    #     f0,
    #     f1,
    #     np.concatenate([np.stack(df_stem.feature.to_numpy()),
    #  np.stack(df_iloop.feature.to_numpy()),
    #  np.stack(df_hloop.feature.to_numpy())], axis=0),
    # ], axis=0)
    features = np.concatenate([
        f0,
        f1], axis=0)
    if len(df_stem) > 0:
        features = np.concatenate([features, np.stack(df_stem.feature.to_numpy())], axis=0)
    if len(df_iloop) > 0:
        features = np.concatenate([features, np.stack(df_iloop.feature.to_numpy())], axis=0)
    if len(df_hloop) > 0:
        features = np.concatenate([features, np.stack(df_hloop.feature.to_numpy())], axis=0)
    return features, target_idx, bb_overlap


df = pd.read_pickle('data/rand_s1_bb_0p1.pkl.gz')


result = []
for _, row in tqdm(df.iterrows()):  # FIXME debug
    features, target_idx, bb_overlap = process_row(row)
    result.append({
        'seq': row['seq'],
        'features': features,
        'target_idx': target_idx,
        'bb_overlap': bb_overlap,
        'input_len': features.shape[0],   # number of proposed bounding boxes (+2 place-holder)
    })
result = pd.DataFrame(result)
result.to_pickle('data/tmp.pkl.gz', compression='gzip')
