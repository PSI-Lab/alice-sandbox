# TODO move to top level utils.py
import numpy as np


def one_idx2arr(one_idx, l):
    # convert list of tuples
    # [(i1, i2, ...), (j1, j2, ...)]
    # to binary matrix
    # assuming 0-based index
    # assuming i < j (upper triangular)
    # unpack idxes
    pairs = []
    for i, j in zip(one_idx[0], one_idx[1]):
        assert 0 <= i < j <= l, "Input index should be 0-based upper triangular"
        pairs.append((i, j))
    #     pairs.append((i-1, j-1))   # only for PDB? - TODO make all dataset consistent
    x = np.zeros((l, l))
    for i, j in pairs:
        x[i, j] = 1
    return pairs, x


def sort_pairs(pairs):
    # sort pairs (i1, j1), (i2, j2), ..., (ik, jk),....
    # such that ik < jk for all k and ik < ik+1
    pairs = [(i, j) if i < j else (j, i) for i, j in pairs]
    pairs = sorted(pairs)
    return pairs


class Stem(object):
    def __init__(self):
        self.one_idx = []

    def validate(self, one_idx):
        # validate pairs
        # no need to validate if empty, or there is only one base pair
        if len(one_idx) <= 1:
            pass
        else:
            # make sure it's sorted
            assert sorted(one_idx) == one_idx
            # make sure every 2 consecutive pairs (ik, jk) & (ik+1, jk+1) satifies ik+1 = ik + 1 and jk+1 = jk - 1
            assert all([a[0] + 1 == b[0] and a[1] - 1 == b[1] for a, b in zip(one_idx[:-1], one_idx[1:])])

    def add_pair(self, pair):
        assert len(pair) == 2
        assert pair[0] < pair[1]
        # add to current collection
        one_idx = self.one_idx.copy()
        one_idx.append(pair)
        # sort
        one_idx = sorted(one_idx)
        # validate
        self.validate(one_idx)
        # update
        self.one_idx = one_idx

    def bounding_box(self):
        # return location and size of bounding box
        assert self.one_idx == sorted(self.one_idx)
        return self.one_idx[0][0], self.one_idx[-1][1], len(self.one_idx)

    def __repr__(self):
        return "Stem location ({0}, {1}) height {2} width {2}".format(*self.bounding_box())


class StemCollection(object):
    def __init__(self):
        self.stems = []
        self.current_stem = None

    def new(self):
        self.current_stem = Stem()

    def conclude(self):
        if len(self.current_stem.one_idx) > 0:
            self.stems.append(self.current_stem)
        self.current_stem = None

    def is_compatible(self, pair):
        # assuming sorted
        assert len(pair) == 2
        assert pair[0] < pair[1]
        if len(self.current_stem.one_idx) == 0:
            return True
        elif pair[0] == self.current_stem.one_idx[-1][0] + 1 and pair[1] == self.current_stem.one_idx[-1][1] - 1:
            return True
        else:
            return False

    def add_pair(self, pair):
        self.current_stem.add_pair(pair)

    def sort(self):
        raise NotImplementedError


class LocalStructureParser(object):

    def __init__(self, pairs):
        self.pairs = pairs
        self.stems = self.parse_stem()
        self.l_bulges, self.r_bulges, self.internal_loops = self.parse_internal_loop()
        self.hairpin_loops = self.parse_hairpin_loop()

    def bounding_box(self, x, structure_type):
        # returns coordinate of top left corner and box size
        # bounding box includes all closing bases for loop structures
        assert structure_type in ['stem', 'l_bulge', 'r_bulge', 'internal_loop', 'hairpin_loop']
        if structure_type == 'stem':
            a, b, w = x.bounding_box()
            return a, b, w, w
        elif structure_type == 'l_bulge':
            # left bulge is specified by:
            # - a_s: list of unpaired positions on the left side
            # - b1 & b2: the two (closing) paired bases on the right side
            a_s, b1, b2 = x
            x0 = min(a_s) - 1
            y0 = b1
            wx = len(a_s) + 2
            wy = 2  # fixed, this is the side without the bulge, so we only include the 2 closing bases
            return x0, y0, wx, wy
        elif structure_type == 'r_bulge':
            # right bulge is specified by:
            # - bs: list of unpaired positions on the right side
            # - a1 & a2: the two (closing) paired bases on the left side
            bs, a1, a2 = x
            x0 = a1
            y0 = min(bs) - 1
            wx = 2  # fixed, this is the side without the bulge, so we only include the 2 closing bases
            wy = len(bs) + 2
            return x0, y0, wx, wy
        elif structure_type == 'internal_loop':
            # internal loop is specified by:
            # - a1: smallest index of the left side loop position (unpaired)
            # - a2: largest index of the left side loop position (unpaired)
            # - b1: smallest index of the right side loop position (unpaired)
            # - b2: largest index of the right side loop position (unpaired)
            a1, a2, b1, b2 = x
            x0 = a1 - 1
            y0 = b1 - 1
            wx = a2 - a1 + 3
            wy = b2 - b1 + 3
            return x0, y0, wx, wy
        elif structure_type == 'hairpin_loop':
            # hairpin loop is specified by:
            # - idxes: list of indexes in the loop (unpaired)
            idxes = x
            x0 = min(idxes) - 1
            y0 = min(idxes) - 1
            wx = max(idxes) - min(idxes) + 3
            wy = max(idxes) - min(idxes) + 3
            return x0, y0, wx, wy

    def parse_stem(self):
        sc = StemCollection()
        sc.new()
        for pair in self.pairs:
            if sc.is_compatible(pair):
                sc.add_pair(pair)
            else:
                sc.conclude()
                sc.new()
                sc.add_pair(pair)
        sc.conclude()
        return sc

    def parse_internal_loop(self):
        l_bulges = []
        r_bulges = []
        internal_loops = []
        # internal loop and bulge, in between stems
        for s1, s2 in zip(self.stems.stems[:-1], self.stems.stems[1:]):
            # make sure these two stems are not fully connected
            assert not (s1.one_idx[-1][0] + 1 == s2.one_idx[0][0] and s1.one_idx[-1][1] - 1 == s2.one_idx[0][1])
            if s1.one_idx[-1][0] + 1 == s2.one_idx[0][0]:  # i connected
                # check if all idxes on the other side are unpaired -> bulge
                idxes = range(s2.one_idx[0][1] + 1, s1.one_idx[-1][1])
                if all([not self.paired(i, self.pairs) for i in idxes]):
                    r_bulges.append((list(idxes), s1.one_idx[-1][0], s2.one_idx[0][0]))
                    print("bulge(R) {} between stems:\n{}\n{}\n".format(list(idxes), s1, s2))
            elif s1.one_idx[-1][1] - 1 == s2.one_idx[0][1]:  # j connected
                # check if all idxes on the other side are unpaired -> bulge
                idxes = range(s1.one_idx[-1][0] + 1, s2.one_idx[0][0])
                if all([not self.paired(i, self.pairs) for i in idxes]):
                    l_bulges.append((list(idxes), s2.one_idx[0][1], s1.one_idx[-1][1]))
                    print("bulge(R) {} between stems:\n{}\n{}\n".format(list(idxes), s1, s2))
            else:  # neither side connected
                # check if all idxes on both sides are unpaired -> internal loop
                idxes_i = range(s1.one_idx[-1][0] + 1, s2.one_idx[0][0])
                idxes_j = range(s2.one_idx[0][1] + 1, s1.one_idx[-1][1])
                if all([not self.paired(i, self.pairs) for i in list(idxes_i) + list(idxes_j)]):
                    internal_loops.append([min(idxes_i), max(idxes_i), min(idxes_j), max(idxes_j)])
                    print("internal loop {} {} between stems:\n{}\n{}\n".format(list(idxes_i), list(idxes_j), s1, s2))
        return l_bulges, r_bulges, internal_loops

    def parse_hairpin_loop(self):
        hairpin_loops = []
        for s in self.stems.stems:
            # check whether the positions enclosed by the stem are unpaired -> hairpin loop
            idxes = range(s.one_idx[-1][0] + 1, s.one_idx[-1][1])
            if all([not self.paired(i, self.pairs) for i in idxes]):
                hairpin_loops.append(list(idxes))
                print("hairpin loop {} within stem:\n{}\n".format(list(idxes), s))
        return hairpin_loops

    def paired(self, position, pairs):
        paired = False
        for pair in pairs:
            if position == pair[0] or position == pair[1]:
                paired = True
        return paired

#
# def parse_local_structures(pairs):
#
#
#
#     # stems only
#     # sc = StemCollection()
#     # sc.new()
#     # for pair in pairs:
#     #     if sc.is_compatible(pair):
#     #         sc.add_pair(pair)
#     #     else:
#     #         sc.conclude()
#     #         sc.new()
#     #         sc.add_pair(pair)
#     # sc.conclude()
#
#     # l_bulges = []
#     # r_bulges = []
#     # internal_loops = []
#     # hairpin_loops = []
#
#     # find in-between stem local structures:
#     # bulge
#     # internal loop
#
#     # TODO sort stem collection
#
#     # for s1, s2 in zip(sc.stems[:-1], sc.stems[1:]):
#     #     # make sure these two stems are not fully connected
#     #     assert not (s1.one_idx[-1][0] + 1 == s2.one_idx[0][0] and s1.one_idx[-1][1] - 1 == s2.one_idx[0][1])
#     #     if s1.one_idx[-1][0] + 1 == s2.one_idx[0][0]:  # i connected
#     #         # check if all idxes on the other side are unpaired -> bulge
#     #         idxes = range(s2.one_idx[0][1] + 1, s1.one_idx[-1][1])
#     #         if all([not paired(i, pairs) for i in idxes]):
#     #             r_bulges.append((list(idxes), s1.one_idx[-1][0], s2.one_idx[0][0]))
#     #             print("bulge(R) {} between stems:\n{}\n{}\n".format(list(idxes), s1, s2))
#     #     elif s1.one_idx[-1][1] - 1 == s2.one_idx[0][1]:  # j connected
#     #         # check if all idxes on the other side are unpaired -> bulge
#     #         idxes = range(s1.one_idx[-1][0] + 1, s2.one_idx[0][0])
#     #         if all([not paired(i, pairs) for i in idxes]):
#     #             l_bulges.append((list(idxes), s2.one_idx[0][1], s1.one_idx[-1][1]))
#     #             print("bulge(R) {} between stems:\n{}\n{}\n".format(list(idxes), s1, s2))
#     #     else:  # neither side connected
#     #         # check if all idxes on both sides are unpaired -> internal loop
#     #         idxes_i = range(s1.one_idx[-1][0] + 1, s2.one_idx[0][0])
#     #         idxes_j = range(s2.one_idx[0][1] + 1, s1.one_idx[-1][1])
#     #         if all([not paired(i, pairs) for i in list(idxes_i) + list(idxes_j)]):
#     #             internal_loops.append([min(idxes_i), max(idxes_i), min(idxes_j), max(idxes_j)])
#     #             print("internal loop {} {} between stems:\n{}\n{}\n".format(list(idxes_i), list(idxes_j), s1, s2))
#
#     # find single-stem local structure:
#     # hairpin loop
#     # for s in sc.stems:
#     #     # check whether the positions enclosed by the stem are unpaired -> hairpin loop
#     #     idxes = range(s.one_idx[-1][0] + 1, s.one_idx[-1][1])
#     #     if all([not paired(i, pairs) for i in idxes]):
#     #         hairpin_loops.append(list(idxes))
#     #         print("hairpin loop {} within stem:\n{}\n".format(list(idxes), s))
#
#     return sc, l_bulges, r_bulges, internal_loops, hairpin_loops


