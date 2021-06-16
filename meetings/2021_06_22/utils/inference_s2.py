import numpy as np
import pandas as pd
from collections import namedtuple
from utils.util_global_struct import process_bb_old_to_new
from utils.rna_ss_utils import arr2db, one_idx2arr
from utils.misc import add_column
import torch
import torch.nn as nn


class ScoreNetwork(nn.Module):

    def __init__(self, num_filters, filter_width, pooling_size):
        super(ScoreNetwork, self).__init__()

        num_filters = [9] + num_filters
        filter_width = [None] + filter_width
        pooling_size = [None] + pooling_size
        cnn_layers = []
        for i, (nf, fw, psize) in enumerate(zip(num_filters[1:], filter_width[1:], pooling_size[1:])):
            cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1))
            cnn_layers.append(nn.BatchNorm2d(nf))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(psize))
        # extra layers
        cnn_layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        cnn_layers.append(nn.Conv2d(num_filters[-1], 1, kernel_size=1, stride=1))
        self.score_network = nn.Sequential(*cnn_layers)

        self.out = nn.Sigmoid()

    def forward_single(self, x):
        x = self.score_network(x)
        return x

    def forward_pair(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)
        # print(x1, x2)

        out = self.out(x1 - x2)
        return out


BoundingBox = namedtuple("BoundingBox", ['bb_x', 'bb_y', 'siz_x', 'siz_y'])


def stem_bbs2arr(bbs, seq_len):
    # TODO validate there's no conflict!
    one_idx = []
    for bb in bbs:
        for offset in range(bb.siz_x):
            x = bb.bb_x + offset
            y = bb.bb_y - offset
            one_idx.append((x, y))
    # convert to list of 2 tuples
    one_idx = list(zip(*one_idx))

    # convert to arr
    pairs, arr = one_idx2arr(one_idx, seq_len)

    return arr


class ScoreNetworkDataEncoder(object):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self):
        pass

    def _encode_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U',
                                                                                                          '4').replace(
            'N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def tile_and_stack(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        l = x.shape[0]
        x1 = x[:, np.newaxis, :]
        x2 = x[np.newaxis, :, :]
        x1 = np.repeat(x1, l, axis=1)
        x2 = np.repeat(x2, l, axis=0)
        return np.concatenate([x1, x2], axis=2)

    def encode_single(self, seq, bp_arr):
        x = self._encode_seq(seq)
        x = self.tile_and_stack(x)
        bp_arr = bp_arr[:, :, np.newaxis]

        x = np.concatenate([x, bp_arr], axis=2)

        # we want this: channel x H x W
        x = torch.from_numpy(x).float()
        x = x.permute(2, 0, 1)

        return x

    def encode_batch(self, seq, bp_arrs):
        xs = []
        for bp_arr in bp_arrs:
            xs.append(self.encode_single(seq, bp_arr)[np.newaxis, :, :, :])
        xs = torch.cat(xs, axis=0)
        return xs


class Predictor(object):

    def __init__(self, model_ckpt, num_filters, filter_width, pooling_size):
        model = ScoreNetwork(num_filters,
                             filter_width,
                             pooling_size)
        model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu')))
        # set to be in inference mode
        model.eval()
        self.model = model
        self.data_encoder = ScoreNetworkDataEncoder()

    def predict_bb_combos(self, seq, bb_combos):
        # bb_combos: list of list of BoundingBox objs
        # convert bb combo to arr
        bb_arrs = []
        for bb_combo in bb_combos:
            bb_arr = stem_bbs2arr(bb_combo, len(seq))
            bb_arrs.append(bb_arr)
        x = self.data_encoder.encode_batch(seq, bb_arrs)
        yp = self.model.forward_single(x)
        return yp.squeeze().detach().numpy()


# tmp helper function

def process_row_bb_combo(row, topk):
    seq = row.seq
    df_target = process_bb_old_to_new(row.bounding_boxes)
    df_target = df_target[df_target['bb_type'] == 'stem']

    df_stem = pd.DataFrame(row.pred_stem_bb)
    # we use df index, make sure it's contiguous
    assert df_stem.iloc[-1].name == len(df_stem) - 1

    bbs = {}
    for idx, r in df_stem.iterrows():
        bbs[idx] = BoundingBox(bb_x=r['bb_x'],
                               bb_y=r['bb_y'],
                               siz_x=r['siz_x'],
                               siz_y=r['siz_y'])

    target_bbs = []
    for idx, r in df_target.iterrows():
        target_bbs.append(BoundingBox(bb_x=r['bb_x'],
                                      bb_y=r['bb_y'],
                                      siz_x=r['siz_x'],
                                      siz_y=r['siz_y']))

    df_valid_combos = pd.DataFrame(row['bb_combos'])
    # get rid of no structure
    df_valid_combos = df_valid_combos[df_valid_combos['num_bps'] > 0]

    df_valid_combos_topk = df_valid_combos.sort_values(by=['num_bps'], ascending=False)[:topk]

    # get list of bb combos
    bb_combos = []
    for bb_inc in df_valid_combos_topk['bb_inc']:
        bb_combos.append([bbs[i] for i in bb_inc])

    # find target
    target_in_combo = True
    target_bb_inc = []
    for target_bb in target_bbs:
        try:
            target_bb_inc.append(next(i for i, bb in bbs.items() if bb == target_bb))
        except StopIteration:
            target_in_combo = False
            break
    if not target_in_combo:
        target_bb_inc = None
    # topk
    target_in_topk = False
    for bb_inc in df_valid_combos_topk['bb_inc']:
        if target_bb_inc == bb_inc:
            target_in_topk = True

    return seq, df_valid_combos, bb_combos, target_bbs, target_bb_inc, target_in_combo, target_in_topk


