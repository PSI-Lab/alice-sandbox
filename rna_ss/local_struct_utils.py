# TODO move to top level utils.py
import numpy as np


def one_idx2arr(one_idx, l, remove_lower_triangular=False):
    # convert list of tuples
    # [(i1, i2, ...), (j1, j2, ...)]
    # to binary matrix
    # assuming 0-based index
    # assuming i < j (upper triangular)
    # unpack idxes
    pairs = []
    for i, j in zip(one_idx[0], one_idx[1]):
        if remove_lower_triangular and i >= j:
            continue
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
        self.verbose = False  # debug use
        self.pairs = pairs
        self.stems = self.parse_stem()
        self.l_bulges, self.r_bulges, self.internal_loops, self.pesudo_knots = self.parse_internal_loop()
        self.hairpin_loops = self.parse_hairpin_loop()
        self.local_structure_bounding_boxes = self.parse_bounding_boxes()

    def parse_bounding_boxes(self):
        local_structures = []
        # stems
        for x in self.stems.stems:
            x0, y0, wx, wy = self.bounding_box(x, 'stem')
            local_structures.append(((x0, y0), (wx, wy), 'stem'))
        # bulges
        for x in self.l_bulges:
            x0, y0, wx, wy = self.bounding_box(x, 'l_bulge')
            local_structures.append(((x0, y0), (wx, wy), 'bulge'))
        for x in self.r_bulges:
            x0, y0, wx, wy = self.bounding_box(x, 'r_bulge')
            local_structures.append(((x0, y0), (wx, wy), 'bulge'))
        # internal loop
        for x in self.internal_loops:
            x0, y0, wx, wy = self.bounding_box(x, 'internal_loop')
            local_structures.append(((x0, y0), (wx, wy), 'internal_loop'))
        # pesudo knots
        for x in self.pesudo_knots:
            x0, y0, wx, wy = self.bounding_box(x, 'pesudo_knot')
            local_structures.append(((x0, y0), (wx, wy), 'pesudo_knot'))
        # hairpin loop
        for x in self.hairpin_loops:
            x0, y0, wx, wy = self.bounding_box(x, 'hairpin_loop')
            local_structures.append(((x0, y0), (wx, wy), 'hairpin_loop'))
        return local_structures

    def bounding_box(self, x, structure_type):
        # TODO log wanrning validate structure local array
        # returns coordinate of top left corner and box size
        # bounding box includes all closing bases for loop structures
        assert structure_type in ['stem', 'l_bulge', 'r_bulge', 'internal_loop', 'pesudo_knot', 'hairpin_loop']
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
        elif structure_type == 'pesudo_knot':
            # pesudo_knot is specified by:
            # - a & b: start and end of pseudo knot
            a, b = x
            x0 = a
            y0 = a
            wx = b - a + 1
            wy = b - a + 1
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
        pesudo_knots = []
        # internal loop and bulge, in between stems
        for s1, s2 in zip(self.stems.stems[:-1], self.stems.stems[1:]):
            # make sure these two stems are not fully connected
            assert not (s1.one_idx[-1][0] + 1 == s2.one_idx[0][0] and s1.one_idx[-1][1] - 1 == s2.one_idx[0][1])
            # check if this is a pseudo knot
            if s1.one_idx[-1][0] < s2.one_idx[0][0] < s1.one_idx[-1][1] < s2.one_idx[0][1]:
                # if not (s1.one_idx[-1][0] < s2.one_idx[0][0] and s2.one_idx[0][1] < s1.one_idx[-1][1]):
                # flattern all idxes
                idx_all = list(sum(s1.one_idx, ())) + list(sum(s2.one_idx, ()))
                pesudo_knots.append([min(idx_all), max(idx_all)])
                if self.verbose:
                    print("pseudo knot {} {} stems:\n{}\n{}\n".format(min(idx_all), max(idx_all), s1, s2))
            elif s1.one_idx[-1][0] + 1 == s2.one_idx[0][0]:  # i connected
                # check if all idxes on the other side are unpaired -> bulge
                idxes = range(s2.one_idx[0][1] + 1, s1.one_idx[-1][1])
                if all([not self.paired(i, self.pairs) for i in idxes]):
                    r_bulges.append((list(idxes), s1.one_idx[-1][0], s2.one_idx[0][0]))
                    if self.verbose:
                        print("bulge(R) {} between stems:\n{}\n{}\n".format(list(idxes), s1, s2))
            elif s1.one_idx[-1][1] - 1 == s2.one_idx[0][1]:  # j connected
                # check if all idxes on the other side are unpaired -> bulge
                idxes = range(s1.one_idx[-1][0] + 1, s2.one_idx[0][0])
                if all([not self.paired(i, self.pairs) for i in idxes]):
                    l_bulges.append((list(idxes), s2.one_idx[0][1], s1.one_idx[-1][1]))
                    if self.verbose:
                        print("bulge(R) {} between stems:\n{}\n{}\n".format(list(idxes), s1, s2))
            else:  # neither side connected
                # check if all idxes on both sides are unpaired -> internal loop
                idxes_i = range(s1.one_idx[-1][0] + 1, s2.one_idx[0][0])
                idxes_j = range(s2.one_idx[0][1] + 1, s1.one_idx[-1][1])
                if all([not self.paired(i, self.pairs) for i in list(idxes_i) + list(idxes_j)]):
                    internal_loops.append([min(idxes_i), max(idxes_i), min(idxes_j), max(idxes_j)])
                    if self.verbose:
                        print("internal loop {} {} between stems:\n{}\n{}\n".format(list(idxes_i), list(idxes_j), s1, s2))
        return l_bulges, r_bulges, internal_loops, pesudo_knots

    def parse_hairpin_loop(self):
        hairpin_loops = []
        for s in self.stems.stems:
            # check whether the positions enclosed by the stem are unpaired -> hairpin loop
            idxes = range(s.one_idx[-1][0] + 1, s.one_idx[-1][1])
            # skip if empty loop <- rare case
            if len(idxes) == 0:
                continue
            if all([not self.paired(i, self.pairs) for i in idxes]):
                hairpin_loops.append(list(idxes))
                if self.verbose:
                    print("hairpin loop {} within stem:\n{}\n".format(list(idxes), s))
        return hairpin_loops

    def paired(self, position, pairs):
        paired = False
        for pair in pairs:
            if position == pair[0] or position == pair[1]:
                paired = True
        return paired


def make_target(structure_arr, local_structure_bounding_boxes, local_unmask_offset=10):
    # local_unmask_offset: how many pixels to extend both ways for unmasking w.r.t. local structure

    # structure_arr: binary matrix representing structure
    # local_structure_bounding_boxes: generated by LocalStructureParser

    # target values - 5 output
    target_vals = np.zeros_like(structure_arr)
    target_vals = np.repeat(target_vals[:, :, np.newaxis], 5, axis=2)
    # to initialize, set output unit 'not_local_structure' to 1's
    target_vals[:, :, 0] = 1
    # # target mask
    # # start with all masked (1), non-mask will be 0
    # # we'll set local structure to 0
    # # this is useful when we want to randomly sample masked locations
    # # (we can sample a random binary array, set lower triangular to 1, and multiply with this one)
    # target_mask = np.ones_like(structure_arr)

    # target mask
    # start with all masked (0), non-mask will be 1
    # we'll set local structure to 1
    # this is useful when we want to randomly sample masked locations
    # (we can sample a random binary array, set lower triangular to 0, and multiply with this one)
    target_mask = np.zeros_like(structure_arr)

    # set values for the 5 sigmoid outputs (not softmax, since multiple can be set to 1)
    # not_local_structure, stem, internal_loop (include bulges), hairpin loop, is_corner
    for ls in local_structure_bounding_boxes:
        (x0, y0), (wx, wy), name = ls

        # skip pseudo knot for now since it's not really local structure
        if name == 'pesudo_knot':
            continue

        # set output unit 'not_local_structure' to 0's
        target_vals[x0:x0 + wx, y0:y0 + wy, 0] = 0

        # # release mask (hacky for hairpin loop, but we'll fix later)
        # target_mask[x0:x0 + wx, y0:y0 + wy] = 0

        # un-mask (hacky for hairpin loop, but we'll fix later)
        ix1 = max(0, x0 - local_unmask_offset)
        ix2 = min(structure_arr.shape[0], x0 + wx + local_unmask_offset)
        iy1 = max(0, y0 - local_unmask_offset)
        iy2 = min(structure_arr.shape[0], y0 + wy + local_unmask_offset)
        target_mask[ix1:ix2, iy1:iy2] = 0

        # set corresponding output value (also validate data)
        # note that for output unit 'stem', 'internal_loop', 'hairpin loop', more than one can be set to 1
        local_arr = structure_arr[x0:x0 + wx, y0:y0 + wy]

        if name == 'stem':
            # make sure it's all zeros except for off-diagonal (1's)
            np.testing.assert_array_equal(local_arr, np.eye(local_arr.shape[0])[::-1])
            # set output unit 'stem' to 1's
            target_vals[x0:x0 + wx, y0:y0 + wy, 1] = 1
            # for the 2 off diagonal corners, set output unit is_corner to 1
            target_vals[x0, y0 + wy - 1, 4] = 1
            target_vals[x0 + wx - 1, y0, 4] = 1

        if name in ['bulge', 'internal_loop']:
            # make sure it's all zeros except for 2 corners (1's)
            tmp_arr = np.zeros(local_arr.shape)
            tmp_arr[0, -1] = 1
            tmp_arr[-1, 0] = 1
            np.testing.assert_array_equal(local_arr, tmp_arr)
            # set output unit 'internal_loop' to 1's
            target_vals[x0:x0 + wx, y0:y0 + wy, 2] = 1
            # for the 2 off diagonal corners, set output unit is_corner to 1
            target_vals[x0, y0 + wy - 1, 4] = 1  # -1 since this is not range!
            target_vals[x0 + wx - 1, y0, 4] = 1

        if name == 'hairpin_loop':
            # make sure it's all zeros except for top right corner
            tmp_arr = np.zeros(local_arr.shape)
            tmp_arr[0, -1] = 1
            np.testing.assert_array_equal(local_arr, tmp_arr)
            # set output unit 'hairpin_loop' to 1's
            target_vals[x0:x0 + wx, y0:y0 + wy, 3] = 1
            # set top right corner
            target_vals[x0, y0 + wy - 1, 4] = 1

    # # fix mask (re-mask lower triangle)
    # target_mask[np.tril_indices_from(target_mask)] = 1

    # fix mask (re-mask lower triangle)
    target_mask[np.tril_indices_from(target_mask)] = 0

    return target_vals, target_mask

