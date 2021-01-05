"""
Inference utils for S2, using self attn model and constraint-aware greedy sampling
"""
import numpy as np
import pandas as pd
import dgutils.pandas as dgp
# from utils_s2 import Predictor
# from util_global_struct import add_bb_bottom_left, compatible_counts, filter_non_standard_stem, validate_global_struct, LocalStructureBb, OneStepChain
# from util_global_struct import filter_non_standard_stem


# FIXME these functions are from util_global_struct


def filter_non_standard_stem(df, seq):
    # filter out stems with nonstandard base pairing
    # df: df_stem
    # 'bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob', 'bl_x', 'bl_y'
    allowed_pairs = ['AT', 'AU', 'TA', 'UA',
                     'GC', 'CG',
                     'GT', 'TG', 'GU', 'UG']
    df_new = []
    for _, row in df.iterrows():
        bb_x = row['bb_x']
        bb_y = row['bb_y']
        siz_x = row['siz_x']
        siz_y = row['siz_y']

        # FIXME should have been int already...
        assert int(bb_x) == bb_x
        assert int(bb_y) == bb_y
        assert int(siz_x) == siz_x
        assert int(siz_y) == siz_y
        bb_x = int(bb_x)
        bb_y = int(bb_y)
        siz_x = int(siz_x)
        siz_y  =int(siz_y)

        seq_x = seq[bb_x:bb_x+siz_x]
        seq_y = seq[bb_y-siz_y+1:bb_y+1][::-1]
        pairs = ['{}{}'.format(x, y) for x, y in zip(seq_x, seq_y)]
        if all([x in allowed_pairs for x in pairs]):
            df_new.append(row)
    df_new = pd.DataFrame(df_new)
    return df_new


def add_bb_bottom_left(df):
    # add bottom left reference point
    # df: requires columns bb_x, bb_y, siz_x, siz_y

    def add_bl(bb_x, bb_y, siz_x, siz_y):
        x = bb_x + siz_x - 1
        y = bb_y - siz_y + 1
        return x, y

    df = dgp.add_columns(df, ['bl_x', 'bl_y'], ['bb_x', 'bb_y', 'siz_x', 'siz_y'], add_bl)
    return df


def compatible_counts(df1, df2, col1, col2, out_name):
    # join df1 and df2 on col1/col2, count the number of compatible entries
    # this is equivalent to:
    # for each row of df1, find how many rows there are in df2 that's compatible

    if isinstance(col1, str):
        col1 = [col1]
    if isinstance(col2, str):
        col2 = [col2]

    assert out_name not in col1 + col2

    # first aggregate count on df2 using col2
    df2_ct = df2[col2].groupby(col2).size().reset_index(name=out_name)
    # hack col name, to avoid duplication
    df2_ct = df2_ct.rename(columns={a: b for a, b in zip(col2, col1)})
    # join to df1
    df = pd.merge(df1, df2_ct, left_on=col1, right_on=col1, how='outer')
    # replace missing entry with 0 (count)
    df = df.fillna(0)

    return df

#
# def filter_non_standard_stem(df, seq):
#     # filter out stems with nonstandard base pairing
#     # df: df_stem
#     # 'bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob', 'bl_x', 'bl_y'
#     allowed_pairs = ['AT', 'AU', 'TA', 'UA',
#                      'GC', 'CG',
#                      'GT', 'TG', 'GU', 'UG']
#     df_new = []
#     for _, row in df.iterrows():
#         bb_x = row['bb_x']
#         bb_y = row['bb_y']
#         siz_x = row['siz_x']
#         siz_y = row['siz_y']
#         seq_x = seq[bb_x:bb_x+siz_x]
#         seq_y = seq[bb_y-siz_y+1:bb_y+1][::-1]
#         pairs = ['{}{}'.format(x, y) for x, y in zip(seq_x, seq_y)]
#         if all([x in allowed_pairs for x in pairs]):
#             df_new.append(row)
#     df_new = pd.DataFrame(df_new)
#     return df_new


class LocalStructureBb(object):

    def __init__(self, top_right_x, top_right_y, size_x, size_y, bb_id, bb_type):
        self.id = bb_id
        assert bb_type in ['stem', 'iloop', 'hloop']
        self.type = bb_type
        self.tr_x = top_right_x
        self.tr_y = top_right_y
        self.size_x = size_x
        self.size_y = size_y
        # also store bottom left
        self.bl_x = top_right_x + size_x - 1
        self.bl_y = top_right_y - size_y + 1

    def share_top_right_corner(self, another_bb):
        # check if self.top_right == another_bb.bottom_left
        if self.tr_x == another_bb.bl_x and self.tr_y == another_bb.bl_y:
            return True
        else:
            return False

    def share_bottom_left_corner(self, another_bb):
        # check if self.bottom_left == another_bb.top_right
        if self.bl_x == another_bb.tr_x and self.bl_y == another_bb.tr_y:
            return True
        else:
            return False

    def overlaps(self, another_bb):
        raise NotImplementedError

    def bp_conflict(self, another_bb):
        # only makes sense for stem bb comparison
        # where the base pair ranges are in conflict with each other
        # i.e. if another_bb cannot be included if self is included in global structure
        # x range
        x1_1, x1_2 = self.tr_x, self.bl_x
        x2_1, x2_2 = another_bb.tr_x, another_bb.bl_x
        x_range_conflict = (x1_1 <= x2_1 <= x1_2) or (x2_1 <= x1_1 <= x2_2)
        # y range
        y1_1, y1_2 = self.bl_y, self.tr_y
        y2_1, y2_2 = another_bb.bl_y, another_bb.tr_y
        y_range_conflict = (y1_1 <= y2_1 <= y1_2) or (y2_1 <= y1_1 <= y2_2)
        # x & y (this can happen if psedoknot conflict, e.g. stem A (2-5)-(30-33) and stem B (28-30)-(44-46))
        xy_range_conflict = (x1_1 <= y2_1 <= x1_2) or (x2_1 <= y1_1 <= x2_2)
        yx_range_conflict = (y1_1 <= x2_1 <= y1_2) or (y2_1 <= x1_1 <= y2_2)
        return x_range_conflict or y_range_conflict or xy_range_conflict or yx_range_conflict

    def __repr__(self):
        return f"{self.type} {self.id} top right ({self.tr_x}, {self.tr_y}), bottom left ({self.bl_x}, {self.bl_y})"


class OneStepChain(object):

    def __init__(self, bb, next_bb=None):
        self.bb = bb
        self.next_bb = next_bb

    def clear_next_bb(self):
        self.next_bb = None

    def add_next_bb(self, next_bb, validate=True):
        if validate:
            assert self.bb.share_top_right_corner(next_bb)
        if self.next_bb is None:
            self.next_bb = []
        self.next_bb.append(next_bb)

    def __repr__(self):
        if self.next_bb:
            tmp = [x.id for x in self.next_bb]
        else:
            tmp = 'N/A'
        return f"{self.bb}. Next: {tmp}"


# end of duplicated functions



def validate_whitelist(id_bb, picked, remaining, df_info):
    # checks
    assert isinstance(picked, set)
    assert isinstance(remaining, set)
    assert len(picked.intersection(remaining)) == 0
    assert isinstance(df_info, pd.DataFrame)
    
    picked_row = df_info[df_info['id_bb'] == id_bb]
    assert len(picked_row) == 1
    picked_row = picked_row.iloc[0]
    
    # check its whitelist is either in remaining, or have already been picked
    if isinstance(picked_row['whitelist1'], list):
        if not set(picked_row['whitelist1']).intersection(remaining.union(picked)):
            return False
    if isinstance(picked_row['whitelist2'], list):
        if not set(picked_row['whitelist2']).intersection(remaining.union(picked)):
            return False

    return True
        

def pick_one_bb(to_be_picked, picked, remaining, df_info):
    # to_be_picked: set of str, bb IDs
    # picked: set of str, bb IDs
    # remaining: set of str, bb IDs
    # df_info: df with info for each bb, also has whitelist(s) and blacklist
    
    
    # checks
    assert isinstance(to_be_picked, set)
    assert isinstance(picked, set)
    assert isinstance(remaining, set)
    assert to_be_picked.issubset(remaining)
    assert len(to_be_picked.intersection(picked)) == 0
    assert isinstance(df_info, pd.DataFrame)
    
    # subset to_be_picked to be those still in remaining
    to_be_picked = to_be_picked.intersection(remaining)
    
    # subset to to_be_picked
    df_bb = df_info[df_info['id_bb'].isin(to_be_picked)]
    
    # sort by prediction
    df_bb = df_bb.sort_values(by=['pred'], ascending=False)
    
    # keep going until one is selected
    while True:
        # pick the first bb
        picked_row = df_bb.iloc[0]
        id_bb = picked_row['id_bb']

        if validate_whitelist(id_bb, picked, remaining, df_info):
            # return this bb
            return id_bb
        else:  # continue
            # TODO check there's > 1 element left?
            df_bb = df_bb[1:]
    
    # should never happen in practise?
    return None


def add_bb(id_bb, picked, remaining, df_info):
    # add bb, remove blacklist bb from remaining, add bb from whitelist
    
    # checks
    assert id_bb in remaining
    
    # get info
    picked_row = df_info[df_info['id_bb'] == id_bb]
    assert len(picked_row) == 1
    picked_row = picked_row.iloc[0]
    
    # add bb
    picked.add(id_bb)
    remaining.remove(id_bb)
#     print(id_bb)
#     print('picked: ', picked)
#     print('remaining: ', remaining)
    
    # blacklist
    if isinstance(picked_row['blacklist'], list):
        remaining -= set(picked_row['blacklist'])
    
    # whitelist(s)
    if isinstance(picked_row['whitelist1'], list):
        wl = set(picked_row['whitelist1'])
        if len(wl.intersection(picked)) == 0:
            id_wl1 = pick_one_bb(wl, picked, remaining, df_info)
            # need to call add_bb ?! TODO
            picked, remaining = add_bb(id_wl1, picked, remaining, df_info)
    
    if isinstance(picked_row['whitelist2'], list):
        wl = set(picked_row['whitelist2'])
        if len(wl.intersection(picked)) == 0:
            id_wl2 = pick_one_bb(wl, picked, remaining, df_info)
            # need to call add_bb ?! TODO
            picked, remaining = add_bb(id_wl2, picked, remaining, df_info)
    
    return picked, remaining

# FIXME put everything in a class!

def summarize_df(df, m_factor=2, hloop=False):
    # calculate median prob and n_proposal_norm
    # m_factor: max proposal per pixel, this can come from:
    # each pixel predict via softmax and scalar size output
    # each pixel predict topk marginal probability

    def _tmp(siz_x, siz_y, prob):
        prob_median = np.median(prob)
        n_proposal_norm = len(prob) / (m_factor * float(siz_x * siz_y))
        if hloop:
            n_proposal_norm = 2 * n_proposal_norm
        return prob_median, n_proposal_norm

    df = dgp.add_columns(df, ['prob_median', 'n_proposal_norm'],
                         ['siz_x', 'siz_y', 'prob'], _tmp)
    # subset columns
    df = df[['bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob_median', 'n_proposal_norm']]

    # TODO assert equal before converting to int!
    df['bb_x'] = df['bb_x'].astype(int)
    df['bb_y'] = df['bb_y'].astype(int)
    df['siz_x'] = df['siz_x'].astype(int)
    df['siz_y'] = df['siz_y'].astype(int)
    return df


def predict_wrapper(bb_stem, bb_iloop, bb_hloop, discard_ns_stem, min_hloop_size, seq, m_factor, predictor):  # TODO add terminate condition
    """
    preprocess s1 output: bb_stem, bb_iloop, bb_hloop = predictor.predict_bb(seq, threshold, topk)
    """
    # FIXME handle cases where some are None
    df_stem = summarize_df(pd.DataFrame(bb_stem), m_factor)
    df_iloop = summarize_df(pd.DataFrame(bb_iloop), m_factor)
    df_hloop = summarize_df(pd.DataFrame(bb_hloop), m_factor=m_factor, hloop=True)

    if discard_ns_stem:
        n_before = len(df_stem)
        df_stem = filter_non_standard_stem(df_stem, seq)
        print("df_stem base pair pruning, before: {}, after: {}".format(n_before, len(df_stem)))
    # hairpin loop - min size
    if min_hloop_size > 0:
        n_before = len(df_hloop)
        df_hloop = df_hloop[df_hloop['siz_x'] >= min_hloop_size]
        print("df_hloop min size pruning, before: {}, after: {}".format(n_before, len(df_hloop)))

    # add bottom left coord
    df_stem = add_bb_bottom_left(df_stem)
    df_iloop = add_bb_bottom_left(df_iloop)
    df_hloop = add_bb_bottom_left(df_hloop)

    picked_bb, df_data = greedy_sample(df_stem, df_iloop, df_hloop, predictor)
    df_picked = df_data[df_data['id_bb'].isin(picked_bb)][['bb_x', 'bb_y', 'siz_x', 'siz_y', 'pred', 'id_bb']]
    # add bb type (using id, hacky)
    df_picked = dgp.add_column(df_picked, 'bb_type', ['id_bb'], lambda x: x.split('_')[0])

    return df_picked


def greedy_sample(df_stem, df_iloop, df_hloop, predictor):
    # input df columns:
    # 'bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob_median', 'n_proposal_norm'
    
    # cleanup - drop invalid ones based on local constraint #
   
    # for each iloop, check:
    # how many compatible outer stems (stem.bottom_left == iloop.top_right)
    df_iloop_cleanup = compatible_counts(df_iloop, df_stem, col1=['bb_x', 'bb_y'], col2=['bl_x', 'bl_y'], out_name='num_compatible_stem_outer')
    # how many compatible inner stems (stem.top_right == iloop.bottom_left)
    df_iloop_cleanup = compatible_counts(df_iloop_cleanup, df_stem, col1=['bl_x', 'bl_y'], col2=['bb_x', 'bb_y'], out_name='num_compatible_stem_inner')
    # drop those rows without compatible stems on both ends
    df_iloop_cleanup = df_iloop_cleanup[(df_iloop_cleanup['num_compatible_stem_inner'] > 0) & (df_iloop_cleanup['num_compatible_stem_outer'] > 0)]
    
    # for each hloop, check:
    # how many compatible outer stems (stem.bottom_left == iloop.top_right)
    df_hloop_cleanup = compatible_counts(df_hloop, df_stem, col1=['bb_x', 'bb_y'], col2=['bl_x', 'bl_y'],
                                         out_name='num_compatible_stem_outer')
    # drop those rows without compatible stem
    df_hloop_cleanup = df_hloop_cleanup[df_hloop_cleanup['num_compatible_stem_outer'] > 0]
    # drop those not symmetric & across diagonal
    df_hloop_cleanup = df_hloop_cleanup[(df_hloop_cleanup['bb_x'] == df_hloop_cleanup['bl_y']) & (
            df_hloop_cleanup['bb_y'] == df_hloop_cleanup['bl_x']) & (
                                                df_hloop_cleanup['siz_x'] == df_hloop_cleanup['siz_y'])]
    
    
    # add IDs #
    df_stem['id_bb'] = [f'stem_{idx}' for idx in range(len(df_stem))]
    df_iloop_cleanup['id_bb'] = [f'iloop_{idx}' for idx in range(len(df_iloop_cleanup))]
    df_hloop_cleanup['id_bb'] = [f'hloop_{idx}' for idx in range(len(df_hloop_cleanup))]

    # bb objects
    stems = [LocalStructureBb(row['bb_x'], row['bb_y'], row['siz_x'], row['siz_y'], row['id_bb'], 'stem') for
             _, row in df_stem.iterrows()]
    iloops = [LocalStructureBb(row['bb_x'], row['bb_y'], row['siz_x'], row['siz_y'], row['id_bb'], 'iloop') for
              idx, (_, row) in enumerate(df_iloop_cleanup.iterrows())]
    hloops = [LocalStructureBb(row['bb_x'], row['bb_y'], row['siz_x'], row['siz_y'], row['id_bb'], 'hloop') for
              idx, (_, row) in enumerate(df_hloop_cleanup.iterrows())]
    
    # prep for whitelist #
    # find next compatible, start with iloop
    iloop_os_chain = []
    for iloop in iloops:
        iloop_os = OneStepChain(iloop)
        for stem in stems:
            if iloop.share_top_right_corner(stem):
                iloop_os.add_next_bb(stem)
        iloop_os_chain.append(iloop_os)

    # find next compatible, start with stem
    stem_os_chain = []
    for stem in stems:
        stem_os = OneStepChain(stem)
        for iloop in iloops:
            if stem.share_top_right_corner(iloop):
                stem_os.add_next_bb(iloop)
        stem_os_chain.append(stem_os)

    # find next compatible, start with hloop
    hloop_os_chain = []
    for hloop in hloops:
        hloop_os = OneStepChain(hloop)
        for stem in stems:
            if hloop.share_top_right_corner(stem):
                hloop_os.add_next_bb(stem)
        hloop_os_chain.append(hloop_os)

    # prep for blacklist #
    # all pairwise compatibility of stems
    distances = np.zeros((len(stems), len(stems)), dtype=object)
    for i in range(len(stems)):
        for j in range(len(stems)):
            d = stems[i].bp_conflict(stems[j])
            distances[i, j] = d
            distances[j, i] = d
    stem_ids = [x.id for x in stems]
    df_stem_conflict = pd.DataFrame(distances, index=stem_ids, columns=stem_ids)
    
    # set whitelist and blacklist #

    # stem: only blacklist
    df_stem = dgp.add_column(df_stem, 'blacklist', ['id_bb'], 
                             lambda x: df_stem_conflict[x][df_stem_conflict[x]].index.tolist())

    # iloop: whitelist x 2
    # note that since we've pruned bb in the begining, both whitelists should be non-empty
    df_iloop_cleanup['whitelist1'] = None
    df_iloop_cleanup['whitelist2'] = None
    # list 1: top right corner connect to stem
    for x in iloop_os_chain:
        id_iloop = x.bb.id
        if x.next_bb is not None:
            tr_stem_ids = [y.id for y in x.next_bb]
            # setting cell value to be a list - hacky way!
            df_iloop_cleanup.loc[df_iloop_cleanup['id_bb'] == id_iloop, 'whitelist1'] = df_iloop_cleanup.loc[df_iloop_cleanup['id_bb'] == id_iloop, 'whitelist1'].apply(lambda x: tr_stem_ids)

    # list 2: bottom left corner connect to stem
    # doing a bit hack here since we're tracing from stems
    tmp = {x: [] for x in df_iloop_cleanup['id_bb'].tolist()}
    for x in stem_os_chain:
        id_stem = x.bb.id
        if x.next_bb is not None: 
            tr_iloop_ids = [y.id for y in x.next_bb]
            for z in tr_iloop_ids:
                tmp[z].append(id_stem)
    for id_iloop, bl_stem_ids in tmp.items():
        df_iloop_cleanup.loc[df_iloop_cleanup['id_bb'] == id_iloop, 'whitelist2'] = df_iloop_cleanup.loc[df_iloop_cleanup['id_bb'] == id_iloop, 'whitelist2'].apply(lambda x: bl_stem_ids)


    # hloop: whitelist
    df_hloop_cleanup['whitelist1'] = None
    # df_hloop_cleanup['whitelist1'] = df_hloop_cleanup['whitelist1'].astype('object')
    for x in hloop_os_chain:
        id_hloop = x.bb.id
        if x.next_bb is not None:
            tr_stem_ids = [y.id for y in x.next_bb]
            df_hloop_cleanup.loc[df_hloop_cleanup['id_bb'] == id_hloop, 'whitelist1'] = df_hloop_cleanup.loc[df_hloop_cleanup['id_bb'] == id_hloop, 'whitelist1'].apply(lambda x: tr_stem_ids)

    # make prediction #
    pred = predictor.predict(df_stem, df_iloop_cleanup, df_hloop_cleanup)
    pred = pred[0, :, 0].detach().numpy()
    assert len(pred) == len(df_stem) + len(df_iloop_cleanup) + len(df_hloop_cleanup)
    df_stem['pred'] = pred[:len(df_stem)]
    df_iloop_cleanup['pred'] = pred[len(df_stem):len(df_stem) + len(df_iloop_cleanup)]
    df_hloop_cleanup['pred'] = pred[-len(df_hloop_cleanup):]
    df_pred = pd.concat([df_stem, df_iloop_cleanup, df_hloop_cleanup])
    
    # greedy sampling with hard constraints #
    df_tmp = df_pred.copy()
    picked = set()
    remaining = set(df_tmp['id_bb'])

    # sort by pred
    df_tmp = df_tmp.sort_values(by=['pred'], ascending=False)

    while len(remaining) > 0:  # FIXME debug - replace with a more meaningful condition
        # pick bb with max pred
        df_remaining = df_tmp[df_tmp['id_bb'].isin(remaining)]
        id_bb = df_remaining.iloc[0]['id_bb']
    #     print('main loop: ', id_bb)

        # verify that adding it won't violate whitelist constraint
        if validate_whitelist(id_bb, picked, remaining, df_tmp):
            # add this bb (also takes care of blacklist and whitelist)
            picked, remaining = add_bb(id_bb, picked, remaining, df_tmp)
        else: # otherwise remove this and continue
            remaining.remove(id_bb)
    
    return picked, df_tmp
    
