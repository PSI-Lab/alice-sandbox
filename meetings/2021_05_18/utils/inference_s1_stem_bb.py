import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import norm
import torch
import torch.nn as nn
import os
import sys
from .rna_ss_utils import db2pairs, one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb


class SimpleConvNet(nn.Module):
    def __init__(self, num_filters, filter_width, hid_shared, hid_output, dropout):
        super(SimpleConvNet, self).__init__()

        # not used for now
        self.context_left, self.context_right = self.compute_context(filter_width)

        num_filters = [8] + num_filters
        filter_width = [None] + filter_width
        cnn_layers = []
        for i, (nf, fw) in enumerate(zip(num_filters[1:], filter_width[1:])):
            # explicitly pad in the first layer with enough context
            if i == 0:
                cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1,
                                            padding=(self.context_left, self.context_right),
                                            padding_mode='zeros'))
            # do not pad in subsequent layers
            else:
                cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1,
                                            padding=0))

            cnn_layers.append(nn.BatchNorm2d(nf))
            cnn_layers.append(nn.ReLU())
            if dropout > 0:
                cnn_layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # FC layers (shared)
        fc_shared = []
        hid_shared = [num_filters[-1]] + hid_shared
        for i, hid in enumerate(hid_shared[1:]):
            fc_shared.append(nn.Conv2d(hid_shared[i], hid, kernel_size=1))
            fc_shared.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_shared)


        # add output specific hidden layers
        # for now support only one layer
        assert len(hid_output) == 1
        hid_output = hid_output[0]
        hid_fc_last = hid_shared[-1]
        # stem
        self.out_stem_on = nn.Sequential(
            nn.Conv2d(hid_fc_last, hid_output, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hid_output, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # TODO loc share hid?
        self.out_stem_loc_x = nn.Sequential(
            nn.Conv2d(hid_fc_last, hid_output, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hid_output, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_stem_loc_y = nn.Sequential(
            nn.Conv2d(hid_fc_last, hid_output, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hid_output, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )

        self.hid_stem_siz = nn.Sequential(
            nn.Conv2d(hid_fc_last, hid_output, kernel_size=1),
            nn.ReLU(),
        )
        self.out_stem_sm_siz = nn.Sequential(
            nn.Conv2d(hid_output, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_stem_sl_siz = nn.Sequential(
            nn.Conv2d(hid_output, 1, kernel_size=1),
        )

    @staticmethod
    def compute_context(filter_width):
        c = np.sum([x - 1 for x in filter_width])
        context_left = c // 2
        context_right = c - context_left
        return context_left, context_right

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc(x)

        # stem
        y_stem_on = self.out_stem_on(x)
        y_stem_loc_x = self.out_stem_loc_x(x)
        y_stem_loc_y = self.out_stem_loc_y(x)
        # y_stem_siz = self.out_stem_siz(x)
        hid_stem_siz = self.hid_stem_siz(x)
        y_stem_sm_siz = self.out_stem_sm_siz(hid_stem_siz)
        y_stem_sl_siz = self.out_stem_sl_siz(hid_stem_siz)

        # collect
        y = {
            # stem
            'stem_on': y_stem_on,
            'stem_location_x': y_stem_loc_x,
            'stem_location_y': y_stem_loc_y,
            # 'stem_size': y_stem_siz,
            'stem_sm_size': y_stem_sm_siz,
            'stem_sl_size': y_stem_sl_siz,

        }

        return y


class SeqPairEncoder(object):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self, x1, x2):
        # x1 & x2: sequence, for now require to be same length (caller should pad with N if needed)
        assert len(x2) == len(x2), "For now require two seqs of same length. Please pad with N."
        x1 = x1.upper().replace('U', 'T')
        x2 = x2.upper().replace('U', 'T')
        assert set(x1).issubset(set(list('ACGTN')))
        assert set(x2).issubset(set(list('ACGTN')))
        self.x1 = x1
        self.x2 = x2
        # encode
        self.x1_1d = self.encode_seq(self.x1)
        self.x2_1d = self.encode_seq(self.x2)
        self.x_2d = self.encode_x(self.x1_1d, self.x2_1d)
        self.x_torch = self.encode_torch_input(self.x_2d)

    def encode_seq(self, x):
        seq = x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def encode_x(self, x1, x2):
        # outer product
        assert len(x1.shape) == 2
        assert x1.shape[1] == 4
        assert len(x2.shape) == 2
        assert x2.shape[1] == 4
        l1 = x1.shape[0]
        l2 = x2.shape[0]
        # print(l1, l2)
        x1 = x1[:, np.newaxis, :]
        x2 = x2[np.newaxis, :, :]
        # print(x1.shape, x2.shape)
        x1 = np.repeat(x1, l2, axis=1)
        x2 = np.repeat(x2, l1, axis=0)
        # print(x1.shape, x2.shape)
        return np.concatenate([x1, x2], axis=2)

    def encode_torch_input(self, x):
        # add batch dim
        assert len(x.shape) == 3
        x = x[np.newaxis, :, :, :]
        # convert to torch tensor
        x = torch.from_numpy(x).float()
        # reshape: batch x channel x H x W
        x = x.permute(0, 3, 1, 2)
        return x


class DataEncoder(object):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self, x, y=None, bb_ref='top_right'):
        # x: sequence
        # y (optional): one_idx (two lists), zero-based
        x = x.upper().replace('U', 'T')
        assert set(x).issubset(set(list('ACGTN'))), "Seq {}, set: {}".format(x, set(x))
        self.x = x
        if y:
            assert len(y) == 2
            assert len(y[0]) == len(y[1])
            assert max(y[0]) < len(x)
            assert max(y[1]) < len(x)
            self.y = y
        assert bb_ref in ['top_left', 'top_right']
        self.bb_ref = bb_ref
        # encode
        self.x_1d = self.encode_seq(self.x)
        self.x_2d = self.encode_x(self.x_1d)
        self.x_torch = self.encode_torch_input(self.x_2d)
        if y:
            self.y_bb, self.y_arrs = self.encode_y(self.y, len(self.x))


    def encode_seq(self, x):
        seq = x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def encode_x(self, x):
        # outer product
        # tile and stack
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        l = x.shape[0]
        x1 = x[:, np.newaxis, :]
        x2 = x[np.newaxis, :, :]
        x1 = np.repeat(x1, l, axis=1)
        x2 = np.repeat(x2, l, axis=0)
        return np.concatenate([x1, x2], axis=2)

    def encode_torch_input(self, x):
        # add batch dim
        assert len(x.shape) == 3
        x = x[np.newaxis, :, :, :]
        # convert to torch tensor
        x = torch.from_numpy(x).float()
        # reshape: batch x channel x H x W
        x = x.permute(0, 3, 1, 2)
        return x

    def encode_y(self, y, l):
        pairs, structure_arr = one_idx2arr(y, l, remove_lower_triangular=True)
        parser = LocalStructureParser(pairs)
        bounding_boxes = parser.parse_bounding_boxes()  # top left corner
        # extra data (needed for plot)
        target_stem_on, target_iloop_on, target_hloop_on, \
        mask_stem_on, mask_iloop_on, mask_hloop_on, \
        target_stem_location_x, target_stem_location_y, target_iloop_location_x, target_iloop_location_y, \
        target_hloop_location_x, target_hloop_location_y, \
        target_stem_size, target_iloop_size_x, target_iloop_size_y, target_hloop_size, \
        mask_stem_location_size, mask_iloop_location_size, \
        mask_hloop_location_size = make_target_pixel_bb(structure_arr, bounding_boxes)
        # only pick the useful ones
        # organize
        y = {
            # stem
            'stem_on': target_stem_on[:, :],
        #     'stem_location_x': torch.from_numpy(target_stem_location_x[:, :, np.newaxis]).float(),
        # # add singleton dimension (these are integer index of softmax index)
        #     'stem_location_y': torch.from_numpy(target_stem_location_y[:, :, np.newaxis]).float(),
        #     'stem_size': torch.from_numpy(target_stem_size[:, :, np.newaxis]).float(),

            # iloop
            'iloop_on': target_iloop_on,
            # 'iloop_location_x': torch.from_numpy(target_iloop_location_x[:, :, np.newaxis]).float(),
            # 'iloop_location_y': torch.from_numpy(target_hloop_location_y[:, :, np.newaxis]).float(),
            # 'iloop_size_x': torch.from_numpy(target_iloop_size_x[:, :, np.newaxis]).float(),
            # 'iloop_size_y': torch.from_numpy(target_iloop_size_y[:, :, np.newaxis]).float(),

            # hloop
            'hloop_on': target_hloop_on,
            # 'hloop_location_x': torch.from_numpy(target_hloop_location_x[:, :, np.newaxis]).float(),
            # 'hloop_location_y': torch.from_numpy(target_hloop_location_y[:, :, np.newaxis]).float(),
            # 'hloop_size': torch.from_numpy(target_hloop_size[:, :, np.newaxis]).float(),
        }
        if self.bb_ref == 'top_right':
            bbs = []
            for (x0, y0), (wx, wy), bb_name in bounding_boxes:
                bbs.append(((x0, y0 + wy - 1), (wx, wy), bb_name))
            return bbs, y
        else:
            return bounding_boxes, y


class Predictor(object):

    def __init__(self, model_ckpt, num_filters=None, filter_width=None, hid_shared=None, hid_output=None, dropout=None):
        model = SimpleConvNet(num_filters=num_filters,
                              filter_width=filter_width,
                              hid_shared=hid_shared,
                              hid_output=hid_output,
                              dropout=dropout)
        model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu')))
        # set to be in inference mode
        model.eval()
        self.model = model

    @staticmethod
    def predict_bounding_box(pred_on, pred_loc_x, pred_loc_y,
                             pred_sm_siz, pred_sl_siz, thres=0.1):

        def _make_mask(l):
            m = np.ones((l, l))
            m[np.tril_indices(l)] = 0
            return m

        def _update(bb_x, bb_y, siz, p_on, p_other, proposed_boxes, pred_box, bb_source):
            assert bb_source in ['sm', 'sl']
            proposed_boxes.append({
                'bb_x': bb_x,
                'bb_y': bb_y,
                'siz_x': siz,
                'siz_y': siz,
                'prob_on_{}'.format(bb_source): p_on,
                'prob_other_{}'.format(bb_source): p_other,
            })
            # set value in pred box, be careful with out of bound index
            x0 = bb_x
            y0 = bb_y - siz + 1  # 0-based
            wx = siz
            wy = siz
            ix0 = max(0, x0)
            iy0 = max(0, y0)
            ix1 = min(x0 + wx, pred_box.shape[0])
            iy1 = min(y0 + wy, pred_box.shape[1])
            pred_box[ix0:ix1, iy0:iy1] = 1
            return proposed_boxes, pred_box

        def sm_top_one(p_on, pred_loc_x, pred_loc_y, pred_sm_siz, i, j):
            loc_x = np.argmax(pred_loc_x[:, i, j])
            loc_y = np.argmax(pred_loc_y[:, i, j])
            # softmax size
            sm_siz = np.argmax(pred_sm_siz[:, i, j]) + 1

            if loc_x == pred_loc_x.shape[0] - 1 or loc_y == pred_loc_y.shape[0] - 1 or sm_siz == pred_sm_siz.shape[
                0]:
                # discard if any argmax = last_one (the "catch-all" unit)
                return []
            else:
                # prob of location
                prob_1 = softmax(pred_loc_x[:, i, j])[loc_x] * softmax(pred_loc_y[:, i, j])[loc_y]
                # softmax: compute joint probability of taking the max value
                prob_sm = prob_1 * softmax(pred_sm_siz[:, i, j])[sm_siz - 1]
                bb_x = i - loc_x
                bb_y = j + loc_y
                # list of one tuple
                result = [(bb_x, bb_y, sm_siz, p_on, prob_sm)]
                return result

        def sl_top_one(p_on, pred_loc_x, pred_loc_y, pred_sl_siz, i, j):
            loc_x = np.argmax(pred_loc_x[:, i, j])
            loc_y = np.argmax(pred_loc_y[:, i, j])

            if loc_x == pred_loc_x.shape[0] - 1 or loc_y == pred_loc_y.shape[0] - 1:
                # discard if any argmax = last_one (the "catch-all" unit)
                return []
            else:
                # scalar size, round to int
                sl_siz = int(np.round(pred_sl_siz[i, j]))
                # avoid setting size 0 or negative # TODO adding logging warning
                if sl_siz < 1:
                    sl_siz = 1
                # # prob of on/off & location
                # prob_1 = p_on * softmax(pred_loc_x[:, i, j])[loc_x] * softmax(pred_loc_y[:, i, j])[loc_y]
                # prob of location
                prob_1 = p_on * softmax(pred_loc_x[:, i, j])[loc_x] * softmax(pred_loc_y[:, i, j])[loc_y]
                # top right corner
                bb_x = i - loc_x
                bb_y = j + loc_y
                # list of one tuple
                result = [(bb_x, bb_y, sl_siz, p_on, prob_1)]
                return result

        # remove singleton dimensions
        pred_on = np.squeeze(pred_on)
        pred_loc_x = np.squeeze(pred_loc_x)
        pred_loc_y = np.squeeze(pred_loc_y)
        pred_sm_siz = np.squeeze(pred_sm_siz)
        pred_sl_siz = np.squeeze(pred_sl_siz)

        # TODO assert on input shape

        # hard-mask
        # note that we're also supporting case where pred_on is not square matrix (i.e. two input seqs are of different length)
        seq_len = pred_on.shape[1]
        m = _make_mask(seq_len)
        # apply mask (for pred, only apply to pred_on since our processing starts from that array)
        pred_on = pred_on * m
        # binary array with all 0's, we'll set the predicted bounding box region to 1
        # this will be used to calculate 'sensitivity'
        pred_box = np.zeros_like(pred_on)
        # also save box locations and probabilities
        proposed_boxes = []

        for i, j in np.transpose(np.where(pred_on > thres)):  # TODO vectorize
            # # save sm box # TODO some computation is duplicated in sm/sl
            result = sm_top_one(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sm_siz, i, j)

            for bb_x, bb_y, sm_siz, prob_on, prob_sm in result:
                # ignore out of bound bbs # TODO print warning?
                if not (0 <= bb_x <= seq_len and 0 <= bb_y <= seq_len):
                    continue
                proposed_boxes, pred_box = _update(bb_x, bb_y, sm_siz, prob_on, prob_sm, proposed_boxes,
                                                   pred_box,
                                                   bb_source='sm')

            # save sl box (it's ok if sl box is identical with sm box, since probabilities will be aggregated in the end)
            result = sl_top_one(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sl_siz, i, j)

            for bb_x, bb_y, sl_siz, prob_on, prob_sl in result:
                # assert 0 <= bb_x <= seq_len
                # assert 0 <= bb_y <= seq_len
                # ignore out of bound bbs # TODO print warning?
                if not (0 <= bb_x <= seq_len and 0 <= bb_y <= seq_len):
                    continue
                proposed_boxes, pred_box = _update(bb_x, bb_y, sl_siz, prob_on, prob_sl, proposed_boxes,
                                                   pred_box,
                                                   bb_source='sl')

        # apply hard-mask to pred box
        pred_box = pred_box * m
        return proposed_boxes, pred_box

    def _nn_pred_to_bb(self, seq, yp, threshold, mask=None):
        # apply mask (if specified)
        # mask is applied to *_on output, masked entries set to 0 (thus those pixels won't predict anything)
        if mask is not None:
            stem_on = yp['stem_on'] * mask
        else:
            stem_on = yp['stem_on']
        # bb
        pred_bb_stem, pred_box_stem = self.predict_bounding_box(pred_on=stem_on, pred_loc_x=yp['stem_location_x'],
                                                                pred_loc_y=yp['stem_location_y'],
                                                                pred_sm_siz=yp['stem_sm_size'],
                                                                pred_sl_siz=yp['stem_sl_size'],
                                                                thres=threshold)

        return yp, pred_bb_stem, pred_box_stem

    def _unique_bbs(self, pred_bb_stem):
        def uniq_boxes(pred_bb):
            # pred_bb: list
            # group rows correspond to the same bb
            # note that each row has only one of the values: prob_sm/ prob_sl
            # we would like to summerize each into a list, so dropping the NaN rows (for each independently)
            df = pd.DataFrame(pred_bb)
            data = df.groupby(by=['bb_x', 'bb_y', 'siz_x', 'siz_y'], as_index=False).agg(lambda x:  x.dropna().tolist()).to_dict('records')
            return data

        if len(pred_bb_stem) > 0:
            uniq_stem = uniq_boxes(pred_bb_stem)
        else:
            uniq_stem = None

        return uniq_stem

    def predict_bb(self, seq, threshold, seq2=None, mask=None):

        # if seq2 specified, predict bb for RNA-RNA
        if seq2:
            de = SeqPairEncoder(seq, seq2)
        else:
            de = DataEncoder(seq)
        # yp = self.model(torch.tensor(de.x_torch))
        yp = self.model(de.x_torch.clone().detach())
        # single example, remove batch dimension
        yp = {k: v.detach().cpu().numpy()[0, :, :, :] for k, v in yp.items()}

        yp, pred_bb_stem, pred_box_stem = self._nn_pred_to_bb(
            seq, yp, threshold, mask)

        uniq_stem = self._unique_bbs(pred_bb_stem)
        return uniq_stem
