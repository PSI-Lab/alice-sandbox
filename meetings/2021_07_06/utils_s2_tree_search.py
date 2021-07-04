import numpy as np
import pandas as pd
import itertools
from collections import namedtuple


BoundingBox = namedtuple("BoundingBox", ['bb_x', 'bb_y', 'siz_x', 'siz_y'])


def range_overlap(r1, r2):
    if r2[0] < r1[1] <= r2[1] or r1[0] < r2[1] <= r1[1]:
        return True
    else:
        return False


def bb_conflict(bb1, bb2):
    r11 = (bb1.bb_x, bb1.bb_x + bb1.siz_x)
    r12 = (bb1.bb_y - bb1.siz_y, bb1.bb_y)
    r21 = (bb2.bb_x, bb2.bb_x + bb2.siz_x)
    r22 = (bb2.bb_y - bb2.siz_y, bb2.bb_y)
    if range_overlap(r11, r21) or range_overlap(r11, r22) or range_overlap(r12, r21) or range_overlap(r12, r22):
        return True
    else:
        return False


def check_bb_compatibility(bb_assignment, bb_idx_to_add, bb_conf_arr):
    bb_inc = np.where(bb_assignment)[0]
    assert bb_idx_to_add not in bb_inc
    # FIXME naive way of checking
    for bbi in bb_inc:
        if bb_conf_arr[bbi, bb_idx_to_add] == 1:
            return False
    return True


class Node:

    def __init__(self, bb_idx, bb_assignment):
        self.bb_idx = bb_idx  # current bb idx
        self.bb_assignment = bb_assignment  # dict of bb assignment (including assignment of current one)
        self.left = None
        self.right = None

    def __repr__(self):
        return f"bb_idx {self.bb_idx}, bb_assignment {self.bb_assignment}"


class StemBbTree:

    def __init__(self, bbs, bb_conf_arr):
        self.bbs = bbs
        self.bb_conf_arr = bb_conf_arr
        self.tree_depth = len(self.bbs)
        self.root_node = self.tree_init()
        self.grow_the_tree(self.root_node)
        # get all leaves TODO better implementation?
        self.leaves = []
        self.get_all_leaves(self.root_node)  # should only be called once!

    def tree_init(self):
        # make the root node
        bb_assignment = np.zeros(len(self.bbs))
        root_node = Node(bb_idx=-1, bb_assignment=bb_assignment)  # a dummy node
        # right node: assign 0 to node 0
        root_node.right = Node(bb_idx=0, bb_assignment=bb_assignment)
        # left: assign 1 to node 0
        new_bb_assignment = bb_assignment.copy()
        new_bb_assignment[0] = 1
        root_node.left = Node(bb_idx=0, bb_assignment=new_bb_assignment)
        return root_node

    def grow_node(self, node):
        # grow immediate children from a node, as long as there is no conflict
        # otherwise the child node with conflict will be set to None

        # if we're already at the leaf, set both children to None
        if node.bb_idx == self.tree_depth - 1:
            node.left = None
            node.right = None
        else:
            new_bb_idx = node.bb_idx + 1
            #         print(new_bb_idx)
            # left: we'll be set bb with bb_idx+1 to 1
            # check if there's any conflict
            if check_bb_compatibility(node.bb_assignment, new_bb_idx, self.bb_conf_arr):
                new_bb_assignment = node.bb_assignment.copy()
                new_bb_assignment[new_bb_idx] = 1
                new_node = Node(new_bb_idx, new_bb_assignment)
                node.left = new_node
            # right: we'll be set bb with bb_idx+1 to 0, this is always doable
            new_bb_assignment = node.bb_assignment.copy()
            new_node = Node(new_bb_idx, new_bb_assignment)
            node.right = new_node
        return node

    def grow_the_tree(self, node):
        if node.left is None and node.right is None:
            node = self.grow_node(node)

        # First recur on left child, if present
        if node.left is not None:
            self.grow_the_tree(node.left)

        # now recur on right child, if present
        if node.right is not None:
            self.grow_the_tree(node.right)

    def get_all_leaves(self, root):
        # If node is leaf node, return it
        if root.left is None and root.right is None:
            self.leaves.append(root)

        # If left child exists,
        # check for leaf recursively
        if root.left is not None:
            self.get_all_leaves(root.left)

        # If right child exists,
        # check for leaf recursively
        if root.right is not None:
            self.get_all_leaves(root.right)


