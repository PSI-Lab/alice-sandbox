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
        target_mask[ix1:ix2, iy1:iy2] = 1

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


def make_target_pixel_bb(structure_arr, local_structure_bounding_boxes, local_unmask_offset=10):
    """
    Most of the time, each pixel can be uniquely assigned to one bounding box.
    In the case of closing pair of a loop, it's assigned to both the stem and the loop.
    In the rare case where the stem is of length 1, and the stem has 2 loops, one on each side,
    the pixel is assigned to 2 loops.
    Thus, it can be observed that each pixel can be assigned to:
        - 0 or 1 stem box
        - 0, 1 or 2 internal loop box (we'll ignore the case of 2 internal loop for now since it's rare FIXME)
        - 0 or 1 hairpin loop
    From the above, we conclude that for each pixel we need at most 4 bounding boxes with unique types
    (of course each box can be turned on/off independently, like in CV):
        - stem box
        - internal loop box 1
        - internal loop box 2 (rarely used, ignored for now)
        - hairpin loop box
    Using the above formulation, we only need to predict the on/off of each box (sigmoid),
    without the need to predict its type (also avoid problem of multiple box permutation).

    To encode the location, we use the top right corner as the reference point,
    and calculate the distance of the current pixel to the reference,
    both horizontally and vertically.
    The idea is to predict it using a softmax over finite classes,
    Horizontal distance (y/columns) <= 0, e.g. 0, -1, ..., -9, -10, -10_beyond.
    Vertical distance (x/rows) >= 0, e.g. 0, 1, ..., 9, 10, 10_beyond.
    Basically we assign one class for each integer distance until some distance away (10in the above example).
    Alternatively we can use one sigmoid unit to encode the direction, and 12-softmax for the magnitude
    (although in the case of 0, direction needs to be masked).

    To encode the size, we use different number of softmax, depending on the box type:
        - stem: one softmax over 1, ..., 9, 10, 10_beyond, since it's square shaped
        - internal loop: two softmax over 1, ..., 9, 10, 10_beyond, one for height one for width
        - hairpin loop: one softmax over 1, ..., 9, 10, 10_beyond, since it's triangle/square shaped
    To account for large sized bb's, also use the scalar valued size directly.

    multiple target & multiple masks
    """

    # TODO to save memory, consider using dtypes e.g. np.uint8

    # binary indicator for each box type & mask
    target_stem_on = np.zeros_like(structure_arr)
    target_iloop_on = np.zeros_like(structure_arr)
    target_hloop_on = np.zeros_like(structure_arr)
    mask_stem_on = np.zeros_like(structure_arr)
    mask_iloop_on = np.zeros_like(structure_arr)
    mask_hloop_on = np.zeros_like(structure_arr)
    # location softmax & mask
    # since we're using the top right corner as the reference point
    # horizontal (y/col) distance: 0, -1, -2, ..., -10, -10_beyond
    # vertical (x/row) distance: 0, +1, +2, ...., +10, +10_beyond
    # to save memory, we store integer encoding (not 1-hot)
    target_stem_location_x = np.zeros_like(structure_arr)
    target_stem_location_y = np.zeros_like(structure_arr)
    target_iloop_location_x = np.zeros_like(structure_arr)
    target_iloop_location_y = np.zeros_like(structure_arr)
    target_hloop_location_x = np.zeros_like(structure_arr)
    target_hloop_location_y = np.zeros_like(structure_arr)
    # size softmax & mask
    # size softmax order: 1, 2, ..., 10, 10_beyond
    target_stem_sm_size = np.zeros_like(structure_arr)
    target_iloop_sm_size_x = np.zeros_like(structure_arr)
    target_iloop_sm_size_y = np.zeros_like(structure_arr)
    target_hloop_sm_size = np.zeros_like(structure_arr)
    # scalar valued size
    target_stem_sl_size = np.zeros_like(structure_arr)
    target_iloop_sl_size_x = np.zeros_like(structure_arr)
    target_iloop_sl_size_y = np.zeros_like(structure_arr)
    target_hloop_sl_size = np.zeros_like(structure_arr)
    # mask can be used for both location and size (which are either both set or not set)
    mask_stem_location_size = np.zeros_like(structure_arr)
    # mask_iloop_location_size = np.zeros_like(structure_arr)
    mask_iloop_location_size = np.zeros_like(structure_arr)
    mask_hloop_location_size = np.zeros_like(structure_arr)

    # above initialization correspond to: everything off & all being masked

    # not_local_structure, stem, internal_loop (include bulges), hairpin loop, is_corner
    for ls in local_structure_bounding_boxes:
        (x0, y0), (wx, wy), name = ls

        # skip pseudo knot for now since it's not really local structure
        if name == 'pesudo_knot':
            continue

        # un-mask box types (hacky for hairpin loop, but we'll fix later)
        # location & size remain being masked
        ix1 = max(0, x0 - local_unmask_offset)
        ix2 = min(structure_arr.shape[0], x0 + wx + local_unmask_offset)
        iy1 = max(0, y0 - local_unmask_offset)
        iy2 = min(structure_arr.shape[0], y0 + wy + local_unmask_offset)
        mask_stem_on[ix1:ix2, iy1:iy2] = 1
        mask_iloop_on[ix1:ix2, iy1:iy2] = 1
        mask_hloop_on[ix1:ix2, iy1:iy2] = 1

        # extract local binary array from original structure, for validation
        local_arr = structure_arr[x0:x0 + wx, y0:y0 + wy]

        # convenience function for calculating index for location and size

        def get_size_index(x):
            # convert integer size to finite array index
            # 1 -> 0, 2 -> 1, ..... 10 -> 9, 11 -> 10, 12 -> 10
            assert x > 0
            assert int(x) == x
            if x <= 10:
                return x - 1
            else:
                return 10

        def get_location_index(x, direction):
            assert direction in ['x', 'y']
            if direction == 'x':
                assert x >= 0
                # 0, +1, +2, ...., +10, +10_beyond
                # 0 -> 0, 1 -> 1, ...., 10 -> 10, 11 -> 11, 12 -> 11, ...
                if x <= 10:
                    return x
                else:
                    return 11
            elif direction == 'y':
                assert x <= 0
                # 0, +1, +2, ...., +10, +10_beyond
                # 0 -> 0, -1 -> 1, ...., -10 -> 10, -11 -> 11, -12 -> 11, ...
                if abs(x) <= 10:
                    return abs(x)
                else:
                    return 11
            else:  # should never be here
                raise ValueError

        def set_location(x0, y0, wx, wy, arr_location_x, arr_location_y):
            # not the most efficient way to set values, but easier to debug
            for i in range(x0, x0 + wx):
                for j in range(y0, y0 + wy):
                    distance_x = i - x0
                    idx_x = get_location_index(distance_x, direction='x')
                    distance_y = j - (y0 + wy - 1)
                    idx_y = get_location_index(distance_y, direction='y')
                    arr_location_x[i, j] = idx_x
                    arr_location_y[i, j] = idx_y
            return arr_location_x, arr_location_y

        if name == 'stem':
            # make sure it's all zeros except for off-diagonal (1's)
            np.testing.assert_array_equal(local_arr, np.eye(local_arr.shape[0])[::-1])
            # stem_on = 1, for all pixels within this box
            target_stem_on[x0:x0 + wx, y0:y0 + wy] = 1
            # size
            assert wx == wy
            idx_size = get_size_index(wx)
            target_stem_sm_size[x0:x0 + wx, y0:y0 + wy] = idx_size
            target_stem_sl_size[x0:x0 + wx, y0:y0 + wy] = wx
            # location
            target_stem_location_x, target_stem_location_y = set_location(x0, y0, wx, wy, target_stem_location_x,
                                                                          target_stem_location_y)
            # unmask location and size
            mask_stem_location_size[x0:x0 + wx, y0:y0 + wy] = 1

        if name in ['bulge', 'internal_loop']:
            # make sure it's all zeros except for 2 corners (1's)
            tmp_arr = np.zeros(local_arr.shape)
            tmp_arr[0, -1] = 1
            tmp_arr[-1, 0] = 1
            np.testing.assert_array_equal(local_arr, tmp_arr)
            # TODO bottom left corner: do not change its value if already set (it's the top right corner of another iloop!)
            keep_old_val = False
            if target_iloop_on[x0+wx-1, y0] == 1:
                keep_old_val = True
                sm_size_x_old = target_iloop_sm_size_x[x0+wx-1, y0]
                sm_size_y_old = target_iloop_sm_size_y[x0+wx-1, y0]
                sl_size_x_old = target_iloop_sl_size_x[x0 + wx - 1, y0]
                sl_size_y_old = target_iloop_sl_size_y[x0 + wx - 1, y0]
                location_x_old = target_iloop_location_x[x0+wx-1, y0]
                location_y_old = target_iloop_location_y[x0+wx-1, y0]
            # iloop_on = 1, for all pixels within this box
            # set loop 2
            target_iloop_on[x0:x0 + wx, y0:y0 + wy] = 1
            # size
            target_iloop_sm_size_x[x0:x0 + wx, y0:y0 + wy] = get_size_index(wx)
            target_iloop_sm_size_y[x0:x0 + wx, y0:y0 + wy] = get_size_index(wy)
            target_iloop_sl_size_x[x0:x0 + wx, y0:y0 + wy] = wx
            target_iloop_sl_size_y[x0:x0 + wx, y0:y0 + wy] = wy
            if keep_old_val:
                target_iloop_sm_size_x[x0+wx-1, y0] = sm_size_x_old
                target_iloop_sm_size_y[x0+wx-1, y0] = sm_size_y_old
                target_iloop_sl_size_x[x0 + wx - 1, y0] = sl_size_x_old
                target_iloop_sl_size_y[x0 + wx - 1, y0] = sl_size_y_old
            # location
            target_iloop_location_x, target_iloop_location_y = set_location(x0, y0, wx, wy,
                                                                              target_iloop_location_x,
                                                                              target_iloop_location_y)
            if keep_old_val:
                target_iloop_location_x[x0+wx-1, y0] = location_x_old
                target_iloop_location_y[x0+wx-1, y0] = location_y_old
            # unmask location and size
            mask_iloop_location_size[x0:x0 + wx, y0:y0 + wy] = 1

        if name == 'hairpin_loop':
            # make sure it's all zeros except for top right corner
            tmp_arr = np.zeros(local_arr.shape)
            tmp_arr[0, -1] = 1
            np.testing.assert_array_equal(local_arr, tmp_arr)
            # set hloop
            target_hloop_on[x0:x0 + wx, y0:y0 + wy] = 1
            # size
            assert wx == wy
            target_hloop_sm_size[x0:x0 + wx, y0:y0 + wy] = get_size_index(wx)
            target_hloop_sl_size[x0:x0 + wx, y0:y0 + wy] = wx
            # location
            target_hloop_location_x, target_hloop_location_y = set_location(x0, y0, wx, wy,
                                                                            target_hloop_location_x,
                                                                            target_hloop_location_y)
            # unmask location and size
            # hacky for hairpin loop, but we'll fix (re-mask lower triangle) at the end
            mask_hloop_location_size[x0:x0 + wx, y0:y0 + wy] = 1

    # fix mask for hloop(re-mask lower triangle)
    mask_hloop_location_size[np.tril_indices_from(mask_hloop_location_size)] = 0
    mask_stem_on[np.tril_indices_from(mask_stem_on)] = 0
    mask_iloop_on[np.tril_indices_from(mask_iloop_on)] = 0
    mask_hloop_on[np.tril_indices_from(mask_hloop_on)] = 0

    return target_stem_on, target_iloop_on, target_hloop_on, \
           mask_stem_on, mask_iloop_on, mask_hloop_on, \
           target_stem_location_x, target_stem_location_y, target_iloop_location_x, target_iloop_location_y, \
           target_hloop_location_x, target_hloop_location_y, \
           target_stem_sm_size, target_iloop_sm_size_x, target_iloop_sm_size_y, target_hloop_sm_size, \
           target_stem_sl_size, target_iloop_sl_size_x, target_iloop_sl_size_y, target_hloop_sl_size, \
           mask_stem_location_size, mask_iloop_location_size, mask_hloop_location_size
