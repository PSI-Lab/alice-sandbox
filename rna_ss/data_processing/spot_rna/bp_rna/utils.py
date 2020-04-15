

def parse_db(s):
    ap = AllPairs(s)
    ap.parse_db()
    return ap.pairs


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
                if self.bracket_round.is_complete():
                    self.pairs.extend(self.bracket_round.flush())
            elif self.bracket_square.is_compatible(s):
                self.bracket_square.add_s(s, i)
                if self.bracket_square.is_complete():
                    self.pairs.extend(self.bracket_square.flush())
            elif self.bracket_triang.is_compatible(s):
                self.bracket_triang.add_s(s, i)
                if self.bracket_triang.is_complete():
                    self.pairs.extend(self.bracket_triang.flush())
            elif self.bracket_curly.is_compatible(s):
                self.bracket_curly.add_s(s, i)
                if self.bracket_curly.is_complete():
                    self.pairs.extend(self.bracket_curly.flush())
            else:
                raise ValueError("Unrecognized character {} at position {}".format(s, i))

        # check that all groups are empty!!
        for bracket in [self.bracket_round, self.bracket_curly, self.bracket_triang, self.bracket_square]:
            if not bracket.is_empty():
                raise ValueError(
                    "Bracket group {}-{} not symmetric: left {} right {}".format(bracket.left_str, bracket.right_str,
                                                                                 bracket.left, bracket.right))


class PairedBrackets(object):

    def __init__(self, left_str, right_str):
        self.left_str = left_str
        self.right_str = right_str
        self.left = []  # left positions
        self.right = []  # right positions

    def is_empty(self):
        return len(self.left) == 0 and len(self.right) == 0

    def is_compatible(self, s):
        return s in [self.left_str, self.right_str]

    def add_s(self, s, pos):
        if s == self.left_str:
            self.left.append(pos)
        elif s == self.right_str:
            self.right.append(pos)
        else:
            raise ValueError("Expect {} or {} but got {}".format(self.left_str, self.right_str, s))

    def is_complete(self):
        return len(self.left) == len(self.right)

    def flush(self):
        # return pairs and reset
        assert self.is_complete()
        pairs = [(i, j) for i, j in zip(self.left, self.right[::-1])]  # right need to reversed
        self.left = []
        self.right = []
        return pairs

