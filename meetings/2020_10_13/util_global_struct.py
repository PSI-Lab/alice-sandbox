import pandas as pd
import copy
import numpy as np
import dgutils.pandas as dgp
import sys
sys.path.insert(0, '../../rna_ss/')
from local_struct_utils import LocalStructureParser


bb_name_mapping = {
    'hairpin_loop': 'hloop',
    'hloop': 'hloop',
    'internal_loop': 'iloop',
    'iloop': 'iloop',
    'bulge': 'iloop',
    'stem': 'stem',
    # not local ss
    'pesudo_knot': 'pknot',
    'pseudo_knot': 'pknot',
}


def process_bb_old_to_new(old_bb):
    # old_bb: list of:
    # (top_left_x, top_left_y), (siz_x, siz_y), bb_type
    df = []
    for (top_left_x, top_left_y), (siz_x, siz_y), bb_type in old_bb:
        bb_x = top_left_x
        bb_y = top_left_y + siz_y - 1
        df.append({
            'bb_x': bb_x,
            'bb_y': bb_y,
            'siz_x': siz_x,
            'siz_y': siz_y,
            'bb_type': bb_name_mapping[bb_type]
        })
    return pd.DataFrame(df)


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


class GlobalConstraint(object):

    def __init__(self, stems, iloops, hloops):
        raise NotImplementedError

    def conflict(self, bb1, bb2):
        # check bb1 and bb2 are in the known bb's
        # same type?
        # check overlap and bp conflict
        raise NotImplementedError


class FullChain(object):

    def __init__(self, start):
        # make sure start is either stem or hloop
        assert start.type in ['stem', 'hloop']
        self.chain = (start,)
        self.start = start
        self.completed = False
        self.id = None

    def add_bb(self, bb):
        last_bb = self.chain[-1]
        assert last_bb.share_top_right_corner(bb), f"{last_bb} {bb}"
        self.chain = self.chain + (bb,)

    def complete(self, validate=True):
        if validate:
            end = self.chain[-1]
            assert end.type == 'stem'
        self.end = end
        self.completed = True

    def merge_chain(self, another_chain):
        # merge another (completed) chain
        assert another_chain.completed
        # make sure they share the same bb
        assert self.chain[-1] == another_chain.chain[0]
        self.chain = self.chain + another_chain.chain[1:]
        self.end = another_chain.end
        self.completed == True

    def __repr__(self):
        status = "Completed" if self.completed else "Incomplete"
        return f"FullChain {self.id} {self.chain} {status}"
#         return f"FullChain {[x.id for x in self.chain]} {status}"


class ChainIdCounter(object):
    prefix = 'chain_'

    def __init__(self):
        self.ct = 0

    def get_id(self):
        ct = self.ct
        self.ct += 1
        return f"{self.prefix}{ct}"


class GrowChain(object):

    def __init__(self, os_chain):
        self.os_chain = os_chain
        self.full_chains = []
        self.chain_id_counter = ChainIdCounter()

    def run(self, starting_bbs):
        for x in starting_bbs:
            fc = FullChain(x)
            self._grow_chain(fc)

    def _grow_chain(self, chain):
        # look up last item in chain
        this_bb = chain.chain[-1]
        # check its next compatible elements
        next_bbs = self.os_chain[this_bb.id].next_bb
        # finish if empty
        # otherwise add next item & recursion
        if next_bbs is None:
            tmp = copy.copy(chain)
            tmp.complete()
            tmp.id = self.chain_id_counter.get_id()  # assign ID
            self.full_chains.append(tmp)
            return
        else:
            #         print(next_bbs, '\n')
            # if end with stem, make a copy, add to full chain list
            if chain.chain[-1].type == 'stem':
                tmp = copy.copy(chain)
                tmp.complete()
                tmp.id = self.chain_id_counter.get_id()  # assign ID
                self.full_chains.append(tmp)

            #         for i in range(len(os_chain[chain.chain[-1].id].next_bb)):
            for i in range(len(next_bbs)):
                # operate on copy before 'branching out' to avoid messing up with a global 'chain'
                tmp = copy.copy(chain)
                tmp.add_bb(next_bbs[i])
                self._grow_chain(tmp)


def chain_compatible(c1, c2, df_stem_conflict):
    stems_1 = [bb for bb in c1.chain if bb.type =='stem']
    stems_2 = [bb for bb in c2.chain if bb.type =='stem']
    for s1 in stems_1:
        for s2 in stems_2:
            if df_stem_conflict[s1.id][s2.id]:
                return False
    return True


class GrowGlobalStruct(object):

    def __init__(self, df_chain_compatibility):
        self.df_chain_compatibility = df_chain_compatibility
        self.global_structs = []


    def grow_global_struct(self, struct, comp_chains, exhausted_only=False):
        # if exhausted_only is set to True, do not store sub global structure when compatible chain is not depleted

        if exhausted_only is False or (exhausted_only is True and len(comp_chains) == 0):
            # global assembly can terminate any time (even for the empty one)
            self.global_structs.append(copy.copy(struct))

        # if no more chain in the compatible list, return
        if len(comp_chains) == 0:
            return

        # add one chain from the list of compatible ones
        for chain in comp_chains:
            struct_new = copy.copy(struct)  # important to make copy inside loop
            struct_new.append(copy.copy(chain))
            # update compatibility list
            chain_id_compatible = set(self.df_chain_compatibility[chain.id].index[self.df_chain_compatibility[chain.id]].to_list())

            #         print(chain, chain_id_compatible)
            comp_chains = [x for x in comp_chains if x.id in chain_id_compatible]
            #         print([x.id for x in struct_new], chain.id, chain_id_compatible, [x.id for x in comp_chains])
            #         print(comp_chains)
            self.grow_global_struct(copy.copy(struct_new), copy.copy(comp_chains))


def validate_global_struct(global_struct):
    # global_struct: list of chains
    # empty structure is always valid
    if len(global_struct) == 0:
        return True

    # find all stems, collect all base pairs
    all_stems = []
    for chain in global_struct:
        all_stems.extend([x for x in chain.chain if x.type == 'stem'])
    bps = []  # list of (i, j) tuple
    for s in all_stems:
        for i, j in zip(range(s.tr_x, s.bl_x + 1), range(s.bl_y, s.tr_y + 1)[::-1]):
            bps.append((i, j))
    bps = sorted(bps)

    # bps can not contain duplicated pairs
    # this should not happen
    assert len(set(bps)) == len(bps)

    # parse local structure implied by these base pairing
    # FIXME for now skip cases we cannot parse
    try:
        lsp = LocalStructureParser(bps)
        lss = process_bb_old_to_new(lsp.local_structure_bounding_boxes)
    except ValueError:
        return False
    # ignore pseudoknot
    lss = lss[lss['bb_type'].isin(['stem', 'iloop', 'hloop'])]

    # check if the bounding boxes are the same after conversion
    df_before = []
    for chain in global_struct:
        for struct in chain.chain:
            df_before.append({
                'bb_x': int(struct.tr_x),
                'bb_y': int(struct.tr_y),
                'siz_x': int(struct.size_x),
                'siz_y': int(struct.size_y),
                'bb_type': struct.type,
            })
    df_before = pd.DataFrame(df_before)

    # check the two dfs are equal (up to row/col swap)
    # col should be same order, no need to check
    # sort rows
    df_before = df_before.sort_values(by=df_before.columns.tolist())
    lss = lss.sort_values(by=lss.columns.tolist())
    return np.array_equal(df_before.values, lss.values)  # use np so we don't compare index


