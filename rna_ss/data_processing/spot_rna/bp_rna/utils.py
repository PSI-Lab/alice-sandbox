import numpy as np


def arr2db(arr, verbose=False):
    # note that this util is WIP and is not optimized
    # in the sense that it does not make the most efficient use of different bracket symbols
    # thus for long sequence with pseudoknot structure for which these exists an unambiguous dot-bracket notation
    # it is not guarantee that this util can decode it in all cases (todo for the future)

    # take into account pseudoknot
    assert len(arr.shape) == 2
    assert arr.shape[0] == arr.shape[1]
    assert np.all((arr == 0) | (arr == 1))
    assert np.max(np.sum(arr, axis=0)) <= 1
    assert np.max(np.sum(arr, axis=1)) <= 1
    idx_pairs = np.where(arr == 1)
    idx_pairs = list(zip(idx_pairs[0], idx_pairs[1]))   # py3 compatible

    # find different groups of stems

    def _group(x):
        # x should already be sorted
        assert all([a[0] < b[0] for a, b in zip(x[:-1], x[1:])])
        g = []
        for a in x:
            if len(g) == 0:
                g.append(a)
            else:
                # this is the 'normal' nested structure
                # where:
                # left_(i) >= left_(i-1) + 1
                # right_(i) <= right(i-1) - 1
                if a[0] >= g[-1][0] + 1 and a[1] <= g[-1][1] - 1:
                    g.append(a)
                else:
                    yield g
                    g = [a]
        if len(g) > 0:
            yield g

    idx_pair_groups = list(_group(idx_pairs))

    result_ambiguous = False

    if len(idx_pair_groups) == 0:
        return '.' * arr.shape[0], result_ambiguous

    symbol_to_use = [0 for _ in range(len(idx_pair_groups))]  # 0: (), 1: [], 2: {}, 3: <>
    # use special symbol if certain group create pseudoknot against other groups
    # nothing needs to be done if there is only one group
    for idx1 in range(0, len(idx_pair_groups)):
        for idx2 in range(idx1 + 1, len(idx_pair_groups)):
            for pair1 in idx_pair_groups[idx1]:
                for pair2 in idx_pair_groups[idx2]:
                    if pair2[0] < pair1[1] < pair2[1]:  # see if they cross
                        # set the first group to special symbol
                        symbol_to_use[idx1] = max([_s for _i, _s in enumerate(symbol_to_use) if
                                                   _i != idx1]) + 1

    if verbose and max(symbol_to_use) != 0:
        print("Pseudoknot detected.")

    if verbose and max(symbol_to_use) > 3:
        print("More than 4 special groups found, need to recycle some symbols?")

    if max(symbol_to_use) > 3:
        result_ambiguous = True

    # print(symbol_to_use)

    db_str = ['.' for _ in range(len(arr))]
    for idx_group, pair_group in enumerate(idx_pair_groups):
        for _i, _j in pair_group:
            i = min(_i, _j)
            j = max(_i, _j)
            if symbol_to_use[idx_group] % 4 == 0:
                db_str[i] = '('
                db_str[j] = ')'
            elif symbol_to_use[idx_group] % 4 == 1:
                db_str[i] = '['
                db_str[j] = ']'
            elif symbol_to_use[idx_group] % 4 == 2:
                db_str[i] = '{'
                db_str[j] = '}'
            elif symbol_to_use[idx_group] % 4 == 3:
                db_str[i] = '<'
                db_str[j] = '>'
            else:
                raise ValueError  # shouldn't be here

    return ''.join(db_str), result_ambiguous


def db2pairs(s):
    ap = AllPairs(s)
    ap.parse_db()
    return ap.pairs


def pairs2idx(pairs):
    # list of tuples
    # to tuple of 2 lists
    return [x[0] for x in pairs], [x[1] for x in pairs]


def idx2arr(idx, seq_len):
    x = np.zeros((seq_len, seq_len))
    x[idx] = 1
    return x


class AllPairs(object):

    def __init__(self, db_str):
        self.db_str = db_str
        self.pairs = []  # list of tuples, where each tuple is one paired positions (i, j)
        # hold on to all bracket groups
        self.bracket_round = PairedBrackets(left_str='(', right_str=')')
        self.bracket_square = PairedBrackets(left_str='[', right_str=']')
        self.bracket_triang = PairedBrackets(left_str='<', right_str='>')
        self.bracket_curly = PairedBrackets(left_str='{', right_str='}')

    def parse_db(self):
        # parse dot-bracket notation
        for i, s in enumerate(self.db_str):
            # add s into bracket collection, if paired
            # also check if any bracket group is completed, if so, flush
            if s == '.':
                continue
            elif self.bracket_round.is_compatible(s):
                self.bracket_round.add_s(s, i)
                # if self.bracket_round.is_complete():
                #     self.pairs.extend(self.bracket_round.flush())
            elif self.bracket_square.is_compatible(s):
                self.bracket_square.add_s(s, i)
                # if self.bracket_square.is_complete():
                #     self.pairs.extend(self.bracket_square.flush())
            elif self.bracket_triang.is_compatible(s):
                self.bracket_triang.add_s(s, i)
                # if self.bracket_triang.is_complete():
                #     self.pairs.extend(self.bracket_triang.flush())
            elif self.bracket_curly.is_compatible(s):
                self.bracket_curly.add_s(s, i)
                # if self.bracket_curly.is_complete():
                #     self.pairs.extend(self.bracket_curly.flush())
            else:
                raise ValueError("Unrecognized character {} at position {}".format(s, i))

        # check that all groups are empty!!
        bracket_groups = [self.bracket_round, self.bracket_curly, self.bracket_triang, self.bracket_square]
        for bracket in bracket_groups:
            if not bracket.is_empty():
                raise ValueError(
                    "Bracket group {}-{} not symmetric: left stack".format(bracket.left_str, bracket.right_str,
                                                                           bracket.left_stack))

        # collect and sort all pairs
        pairs = []
        for bracket in bracket_groups:
            pairs.extend(bracket.pairs)
        pairs = sorted(pairs)
        self.pairs = pairs


class PairedBrackets(object):

    def __init__(self, left_str, right_str):
        self.left_str = left_str
        self.right_str = right_str
        self.pairs = []   # list of tuples (i, j)
        self.left_stack = []  # left positions

    def is_empty(self):
        return len(self.left_stack) == 0

    def is_compatible(self, s):
        return s in [self.left_str, self.right_str]

    def add_s(self, s, pos):
        if s == self.left_str:
            self.left_stack.append(pos)
        elif s == self.right_str:
            # pop on item from left_stack
            i = self.left_stack.pop()
            self.pairs.append((i, pos))
        else:
            raise ValueError("Expect {} or {} but got {}".format(self.left_str, self.right_str, s))

    # def is_complete(self):
    #     return len(self.left) == len(self.right)
    #
    # def flush(self):
    #     # return pairs and reset
    #     assert self.is_complete()
    #     pairs = [(i, j) for i, j in zip(self.left, self.right[::-1])]  # right need to reversed
    #     # FIXME debug
    #     print('flushing {} {}'.format(self.left_str, self.right_str))
    #     print(pairs)
    #     self.left = []
    #     self.right = []
    #     return pairs

