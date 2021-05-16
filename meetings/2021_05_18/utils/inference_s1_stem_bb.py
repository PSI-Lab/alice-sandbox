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

    @staticmethod
    def filter_non_standard_stem(df, seq):
        # filter out stems with nonstandard base pairing
        # df: df_stem
        # 'bb_x', 'bb_y', 'siz_x', 'siz_y', 'prob', 'bl_x', 'bl_y'
        allowed_pairs = ['AT', 'AU', 'TA', 'UA',
                         'GC', 'CG',
                         'GT', 'TG', 'GU', 'UG']
        df_new = []
        for _, row in df.iterrows():
            bb_x = row['bb_x']
            bb_y = row['bb_y']
            siz_x = row['siz_x']
            siz_y = row['siz_y']
            seq_x = seq[bb_x:bb_x + siz_x]
            seq_y = seq[bb_y - siz_y + 1:bb_y + 1][::-1]
            pairs = ['{}{}'.format(x, y) for x, y in zip(seq_x, seq_y)]
            if all([x in allowed_pairs for x in pairs]):
                df_new.append(row)
        df_new = pd.DataFrame(df_new)
        return df_new

    @staticmethod
    def filter_diagonal_stem(df):
        # for stem, we need the bb bottom left corner to be in the upper triangular (exclude diagonal), i.e.i < j
        df_new = []
        for _, row in df.iterrows():
            bb_x = row['bb_x']
            bb_y = row['bb_y']
            siz_x = row['siz_x']
            siz_y = row['siz_y']
            assert siz_x == siz_y

            bl_x = bb_x + siz_x - 1
            bl_y = bb_y - siz_y + 1

            if bl_x < bl_y:
                df_new.append(row)
            else:
                # debug
                print("Skip stem {}-{} size {}".format(bb_x, bb_y, siz_x))
        df_new = pd.DataFrame(df_new)
        return df_new

    @staticmethod
    def filter_out_of_range_bb(df, l):
        # filter out invalid bounding box that falls out of range
        df_new = []
        for _, row in df.iterrows():
            bb_x = row['bb_x']
            bb_y = row['bb_y']
            siz_x = row['siz_x']
            siz_y = row['siz_y']
            if (0 <= bb_x < l) and (0 <= bb_x + siz_x - 1 < l) and (0 <= bb_y < l) and (0 <= bb_y - siz_y + 1 < l):
                df_new.append(row)
        df_new = pd.DataFrame(df_new)
        return df_new

    def predict_bb(self, seq, threshold, filter_valid=True):

        # # if seq2 specified, predict bb for RNA-RNA
        # if seq2:
        #     de = SeqPairEncoder(seq, seq2)
        # else:
        #     de = DataEncoder(seq)
        # yp = self.model(torch.tensor(de.x_torch))
        de = DataEncoder(seq)
        yp = self.model(de.x_torch.clone().detach())
        # single example, remove batch dimension
        yp = {k: v.detach().cpu().numpy()[0, :, :, :] for k, v in yp.items()}

        yp, pred_bb_stem, pred_box_stem = self._nn_pred_to_bb(
            seq, yp, threshold, mask=None)

        uniq_stem = self._unique_bbs(pred_bb_stem)
        df_stem = pd.DataFrame(uniq_stem)

        # filter out invalid ones
        if filter_valid:
            df_stem = self.filter_out_of_range_bb(df_stem, len(seq))
            df_stem = self.filter_diagonal_stem(df_stem)
            df_stem = self.filter_non_standard_stem(df_stem, seq)

        return df_stem


class Evaluator(object):

    def __init__(self, predictor):
        if predictor is None:
            print("Initializing evaluator without predictor")  # some class methods can still be used
        else:
            assert isinstance(predictor, Predictor)
        self.predictor = predictor
        # hold on to data for one example, for convenience
        self.data_encoder = None
        # predictions
        self.yp = None
        self.pred_bb_stem = None

    @staticmethod
    def make_target_bb_df(target_bb, convert_tl_to_tr=False):
        # convert_tl_to_tr: if set to True, target_bb has top left corner, will be converted to top right corner
        df_target_stem = []
        for (bb_x, bb_y), (siz_x, siz_y), bb_type in target_bb:
            if convert_tl_to_tr:
                bb_y = bb_y + siz_y - 1
            row = {
                'bb_x': bb_x,
                'bb_y': bb_y,
                'siz_x': siz_x,
                'siz_y': siz_y,
            }
            if bb_type == 'stem':
                df_target_stem.append(row)
        if len(df_target_stem) > 0:
            df_target_stem = pd.DataFrame(df_target_stem)

        return df_target_stem

    @staticmethod
    def _calculate_bb_metrics(df_target, df_pred):

        def is_identical(bb1, bb2):
            return bb1 == bb2  # this should work? FIXME
            # bb1_x, bb1_y, siz1_x, siz1_y = bb1
            # bb2_x, bb2_y, siz2_x, siz2_y = bb2
            # # FIXME debug! any off-by-1 error?
            # return abs(bb1_x-bb2_x)<=1 and abs(bb1_y-bb2_y)<=1 and abs(siz1_x-siz2_x)<=1 and abs(siz1_y-siz2_y)<=1

        def is_local_shift(bb1, bb2):
            """check if bb1 and bb2 are local shift/expand version of each other
            max diff <= 1, include case where bb1 == bb2"""
            bb1_x, bb1_y, siz1_x, siz1_y = bb1
            bb2_x, bb2_y, siz2_x, siz2_y = bb2
            max_diff = max(abs(bb1_x - bb2_x), abs(bb1_y - bb2_y), abs(siz1_x - siz2_x), abs(siz1_y - siz2_y))
            if max_diff <= 1:
                return True
            else:
                return False

        def area_overlap(bb1, bb2):
            bb1_x, bb1_y, siz1_x, siz1_y = bb1
            bb2_x, bb2_y, siz2_x, siz2_y = bb2
            # calculate overlap rectangle, check to see if it's empty
            x0 = max(bb1_x, bb2_x)
            x1 = min(bb1_x + siz1_x - 1, bb2_x + siz2_x - 1)  # note this is closed end
            y0 = max(bb1_y - siz1_y + 1, bb2_y - siz2_y + 1)  # closed end
            y1 = min(bb1_y, bb2_y)
            if x1 >= x0 and y1 >= y0:
                # return area of overlapping rectangle
                return (x1 - x0 + 1) * (y1 - y0 + 1)
            else:
                return 0

        assert set(df_target.columns) == {'bb_x', 'bb_y', 'siz_x', 'siz_y'}
        assert set(df_pred.columns) == {'bb_x', 'bb_y', 'siz_x', 'siz_y'}

        # make sure all rows are unique
        assert not df_target.duplicated().any()
        assert not df_pred.duplicated().any()

        # w.r.t. target
        n_target_total = len(df_target)
        n_target_local = 0
        n_target_identical = 0
        n_target_overlap = 0
        n_target_nohit = 0
        for _, row1 in df_target.iterrows():
            bb1 = (row1['bb_x'], row1['bb_y'], row1['siz_x'], row1['siz_y'])
            found_identical = False
            found_local_shift = False
            found_overlapping = False
            best_area_overlap = 0
            best_bb_overlap = None
            for _, row2 in df_pred.iterrows():
                bb2 = (row2['bb_x'], row2['bb_y'], row2['siz_x'], row2['siz_y'])
                if is_identical(bb1, bb2):
                    found_identical = True
                elif is_local_shift(bb1, bb2): # note this is overlapping but NOT local shift due to "elif"
                    found_local_shift = True
                elif area_overlap(bb1, bb2) > 0:  # note this is overlapping but NOT identical or local shift due to "elif"
                    found_overlapping = True
                    this_area = area_overlap(bb1, bb2)
                    if this_area > best_area_overlap:
                        best_area_overlap = this_area
                        best_bb_overlap = bb2
                else:
                    pass
            if found_identical:
                n_target_identical += 1
            elif found_local_shift:
                n_target_local += 1
            elif found_overlapping:
                n_target_overlap += 1
                # debug print closest pred bb
                print("target bb: {}".format(bb1))
                print("best overlapping bb: {}".format(best_bb_overlap))
                print("best overlapping area: {}".format(best_area_overlap))
            else:
                n_target_nohit += 1

        # FIXME there is some wasted comparison here (can be combined with last step)
        # w.r.t. pred
        n_pred_total = len(df_pred)
        n_pred_local = 0
        n_pred_identical = 0
        n_pred_overlap = 0
        n_pred_nohit = 0
        for _, row1 in df_pred.iterrows():
            bb1 = (row1['bb_x'], row1['bb_y'], row1['siz_x'], row1['siz_y'])
            found_identical = False
            found_local_shift = False
            found_overlapping = False
            for _, row2 in df_target.iterrows():
                bb2 = (row2['bb_x'], row2['bb_y'], row2['siz_x'], row2['siz_y'])
                if is_identical(bb1, bb2):
                    found_identical = True
                elif is_local_shift(bb1, bb2): # note this is overlapping but NOT local shift due to "elif"
                    found_local_shift = True
                elif area_overlap(bb1, bb2) > 0:  # note this is overlapping but NOT identical or local shift due to "elif"
                    found_overlapping = True
                else:
                    pass
            if found_identical:
                n_pred_identical += 1
            elif found_local_shift:
                n_pred_local += 1
            elif found_overlapping:
                n_pred_overlap += 1
            else:
                n_pred_nohit += 1
        result = {
            'n_target_total': n_target_total,
            'n_target_local': n_target_local,
            'n_target_identical': n_target_identical,
            'n_target_overlap': n_target_overlap,
            'n_target_nohit': n_target_nohit,
            'n_pred_total': n_pred_total,
            'n_pred_local': n_pred_local,
            'n_pred_identical': n_pred_identical,
            'n_pred_overlap': n_pred_overlap,
            'n_pred_nohit': n_pred_nohit,
        }
        return result

    def calculate_bb_metrics(self, df_target, df_pred):
        if (df_target is None or len(df_target) == 0) and (df_pred is None or len(df_pred) == 0):
            return {
                'n_target_total': 0,
                'n_target_local': 0,
                'n_target_identical': 0,
                'n_target_overlap': 0,
                'n_target_nohit': 0,
                'n_pred_total': 0,
                'n_pred_local': 0,
                'n_pred_identical': 0,
                'n_pred_overlap': 0,
                'n_pred_nohit': 0,
            }

        elif df_target is None or len(df_target) == 0:
            return {
                'n_target_total': 0,
                'n_target_local': 0,
                'n_target_identical': 0,
                'n_target_overlap': 0,
                'n_target_nohit': 0,
                'n_pred_total': len(df_pred),
                'n_pred_local': 0,
                'n_pred_identical': 0,
                'n_pred_overlap': 0,
                'n_pred_nohit': 0,
            }
        elif df_pred is None or len(df_pred) == 0:
            return {
                'n_target_total': len(df_target),
                'n_target_local': 0,
                'n_target_identical': 0,
                'n_target_overlap': 0,
                'n_target_nohit': 0,
                'n_pred_total': 0,
                'n_pred_local': 0,
                'n_pred_identical': 0,
                'n_pred_overlap': 0,
                'n_pred_nohit': 0,
            }
        else:
            return self._calculate_bb_metrics(df_target, df_pred)

    def stem_bb_to_arr(self, df_bb, seq_len):
        # TODO hard mask? (no need?) remove invalid bb?
        arr = np.zeros((seq_len, seq_len))
        for _, row in df_bb.iterrows():
            bb_x, bb_y, siz_x, siz_y = row['bb_x'], row['bb_y'], row['siz_x'], row['siz_y']
            assert siz_x == siz_y
            for offset in range(siz_x):
                i = bb_x + offset
                j = bb_y - offset
                # ignore out-of-bound ones TODO should prune bb before hand!
                if not (0<=i<seq_len and 0<=j<seq_len):
                    continue
                arr[i, j] = 1
        return arr


    def calculate_bp_metrics(self, df_target, df_pred, seq_len):
        # base pair metrics
        arr_target = self.stem_bb_to_arr(df_target, seq_len)
        arr_pred = self.stem_bb_to_arr(df_pred, seq_len)
        sensitivity = np.sum(arr_target * arr_pred)/np.sum(arr_target)
        return sensitivity

    @staticmethod
    def sensitivity_specificity(target_on, pred_box, hard_mask):
        sensitivity = np.sum((pred_box * target_on) * hard_mask) / np.sum(target_on * hard_mask)
        specificity = np.sum((1 - pred_box) * (1 - target_on) * hard_mask) / np.sum((1 - target_on) * hard_mask)
        return sensitivity, specificity

    @staticmethod
    def bb_unique(bb):
        # convert list of bb to unique bb
        # input:
        # [{'bb_x': 0, 'bb_y': 16, 'siz_x': 3, 'siz_y': 3, 'prob': 0.1654079953181778},
        # {'bb_x': 0, 'bb_y': 72, 'siz_x': 8, 'siz_y': 8, 'prob': 0.11180505377843244}, ...]
        if len(bb) > 0:
            df_tmp = pd.DataFrame(bb)
            df_tmp = df_tmp[['bb_x', 'bb_y', 'siz_x', 'siz_y']].drop_duplicates()
            return df_tmp.to_dict('records')
        else:
            return bb
