# df with bb proposal -> df with valid global structs
from util_global_struct import process_bb_old_to_new, add_bb_bottom_left, compatible_counts, LocalStructureBb, OneStepChain, GrowChain, GrowGlobalStruct, FullChain, chain_compatible, validate_global_struct
import numpy as np
import copy
import pandas as pd
import dgutils.pandas as dgp


def prune_bb(df_bb, min_pixel_pred=3, min_prob=0.5):
    # prune initial bounding box prediction
    # bounding box is kept if one if the follow conditions is satisfied:
    # - num_proposal >= 3, or
    # - max(prob) >= 0.5
    df_bb = df_bb[(df_bb['prob'].apply(len) >= min_pixel_pred) | (df_bb['prob'].apply(max) >= min_prob)]
    return df_bb


def format_global_structures(global_struct, df_stem, df_iloop, df_hloop):
    # global_struct: list of chain

    def lookup_prob(bb, df_bb):
        hit = df_bb[(df_bb['bb_x'] == bb.tr_x) & (df_bb['bb_y'] == bb.tr_y) & (df_bb['siz_x'] == bb.size_x) & (
                df_bb['siz_y'] == bb.size_y)]
        assert len(hit) == 1
        return hit.iloc[0]['prob']

    df = []
    for chain in global_struct:
        for bb in chain.chain:
            data = {
                'bb_x': int(bb.tr_x),
                'bb_y': int(bb.tr_y),
                'siz_x': int(bb.size_x),
                'siz_y': int(bb.size_y),
                'bb_type': bb.type,
            }
            if bb.type == 'stem':
                prob = lookup_prob(bb, df_stem)
            elif bb.type == 'iloop':
                prob = lookup_prob(bb, df_iloop)
            elif bb.type == 'hloop':
                prob = lookup_prob(bb, df_hloop)
            else:
                raise ValueError
            data['prob'] = prob
            df.append(data)
    df = pd.DataFrame(df)
    return df


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


def make_bb(row):
    if isinstance(row['bb_stem'], list):  # missing val will be pd.NaN
        df_stem = pd.DataFrame(row['bb_stem'])
        df_stem = prune_bb(df_stem)
        df_stem = add_bb_bottom_left(df_stem)
    else:
        df_stem = pd.DataFrame([], columns=['bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob', 'bl_x', 'bl_y'])

    if isinstance(row['bb_iloop'], list):  # missing val will be pd.NaN
        df_iloop = pd.DataFrame(row['bb_iloop'])
        df_iloop = prune_bb(df_iloop)
        df_iloop = add_bb_bottom_left(df_iloop)
    else:
        df_iloop = pd.DataFrame([], columns=['bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob', 'bl_x', 'bl_y'])

    if isinstance(row['bb_hloop'], list):  # missing val will be pd.NaN
        df_hloop = pd.DataFrame(row['bb_hloop'])
        df_hloop = prune_bb(df_hloop)
        df_hloop = add_bb_bottom_left(df_hloop)
    else:
        df_hloop = pd.DataFrame([], columns=['bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob', 'bl_x', 'bl_y'])
    return df_stem, df_iloop, df_hloop

def generate_structs(df_stem, df_iloop, df_hloop):
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

    # bb objects
    # use enumerate on df since we want to contiguous ids, not the original df index
    stems = [LocalStructureBb(row['bb_x'], row['bb_y'], row['siz_x'], row['siz_y'], f'stem_{idx}', 'stem') for
             idx, (_, row) in enumerate(df_stem.iterrows())]
    iloops = [LocalStructureBb(row['bb_x'], row['bb_y'], row['siz_x'], row['siz_y'], f'iloop_{idx}', 'iloop') for
              idx, (_, row) in enumerate(df_iloop_cleanup.iterrows())]
    hloops = [LocalStructureBb(row['bb_x'], row['bb_y'], row['siz_x'], row['siz_y'], f'hloop_{idx}', 'hloop') for
              idx, (_, row) in enumerate(df_hloop_cleanup.iterrows())]

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

    # for convenience
    os_chain = {x.bb.id: x for x in iloop_os_chain + stem_os_chain + hloop_os_chain}

    grow_chain = GrowChain(os_chain)
    # start with stem or hloop
    grow_chain.run(stems + hloops)

    # all pairwise compatibility of stems
    distances = np.zeros((len(stems), len(stems)), dtype=object)
    for i in range(len(stems)):
        for j in range(len(stems)):
            d = stems[i].bp_conflict(stems[j])
            distances[i, j] = d
            distances[j, i] = d
    stem_ids = [x.id for x in stems]
    df_stem_conflict = pd.DataFrame(distances, index=stem_ids, columns=stem_ids)

    # all pairwise compatibility of chains
    distances = np.zeros((len(grow_chain.full_chains), len(grow_chain.full_chains)), dtype=object)
    for i in range(len(grow_chain.full_chains)):
        for j in range(len(grow_chain.full_chains)):
            d = chain_compatible(grow_chain.full_chains[i], grow_chain.full_chains[j], df_stem_conflict)
            distances[i, j] = d
            distances[j, i] = d
    chain_ids = [x.id for x in grow_chain.full_chains]
    df_chain_compatibility = pd.DataFrame(distances, index=chain_ids, columns=chain_ids)

    grow_global = GrowGlobalStruct(df_chain_compatibility)

    # collect all 'subset strutucres'
    # grow_global.grow_global_struct([], copy.copy(grow_chain.full_chains))
    # greedy approach
    grow_global.grow_global_struct([], copy.copy(grow_chain.full_chains),
                                   exhausted_only=True)  # FIXME not working? idx=180

    valid_global_structs = [x for x in grow_global.global_structs if validate_global_struct(x)]


    gt_found = False
    global_struct_dfs = []
    # reformat data, add in probability
    for gs in valid_global_structs:
        # skip empty
        if len(gs) == 0:
            continue
        df_gs = format_global_structures(gs, df_stem, df_iloop, df_hloop)
        # add in prob count, median
        df_gs = dgp.add_columns(df_gs, ['n_proposal', 'prob_median'], ['prob'], lambda x: (len(x), np.median(x)))
        # add normalized count (count/upper_bound)
        df_gs = dgp.add_column(df_gs, 'n_proposal_norm', ['n_proposal', 'siz_x', 'siz_y'],
                               lambda n, x, y: float(n) / (x * y))
        # drop full prob for printing
        # print(df_gs.drop(columns=['prob']).to_string())
        global_struct_dfs.append(df_gs.drop(columns=['prob']))  # save space

        # as a reference, check whether this is the ground truth
        df_tmp = pd.merge(df_gs[['bb_x', 'bb_y', 'siz_x', 'siz_y', 'bb_type']], df_target, how='inner')
        if len(df_tmp) == len(df_target):
            # print("Ground truth!")
            gt_found = True

        # print('')
    return global_struct_dfs, gt_found


df = pd.read_pickle('../2020_09_22/data/rand_s1_bb_0p1.pkl.gz')
df_out = []

# for debug
err_idxes = []

for idx, row in df.iterrows():
# for idx, row in [(19301, df.iloc[19301])]:
    # debug
    if len(row['seq']) > 60:
        continue

    print(idx)
    try:
        df_target = process_bb_old_to_new(row['bounding_boxes'])
        df_stem, df_iloop, df_hloop = make_bb(row)
        n_bb_found = check_bb_sensitivity(df_target, df_stem, df_iloop, df_hloop)
        global_struct_dfs, gt_found = generate_structs(df_stem, df_iloop, df_hloop)

        row['df_target'] = df_target
        row['n_bb_found'] = n_bb_found
        row['global_struct_dfs'] = global_struct_dfs
        row['gt_found'] = gt_found
        df_out.append(row)
    except Exception as e:
        err_idxes.append((idx, str(e)))

df_out = pd.DataFrame(df_out)
# print(df_out)
df_out.to_pickle('data/rand_s1_bb_0p1_global_structs_60.pkl.gz')

print(err_idxes)



