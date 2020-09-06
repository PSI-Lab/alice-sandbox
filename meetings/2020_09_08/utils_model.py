import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../../rna_ss/')
from utils import db2pairs
from local_struct_utils import one_idx2arr, sort_pairs, LocalStructureParser, make_target_pixel_bb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SimpleConvNet(nn.Module):
    def __init__(self, num_filters, filter_width, dropout):
        super(SimpleConvNet, self).__init__()

        num_filters = [8] + num_filters
        filter_width = [None] + filter_width
        cnn_layers = []
        for i, (nf, fw) in enumerate(zip(num_filters[1:], filter_width[1:])):
            assert fw % 2 == 1  # odd
            cnn_layers.append(nn.Conv2d(num_filters[i], nf, kernel_size=fw, stride=1, padding=fw//2))
            cnn_layers.append(nn.BatchNorm2d(nf))
            cnn_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                cnn_layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # self.fc = nn.Conv2d(num_filters[-1], 5, kernel_size=1)
        self.fc = nn.Conv2d(num_filters[-1], 50, kernel_size=1)
        # FIXME add relu!

        # add output specific hidden layers

        # stem
        self.out_stem_on = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_stem_loc_x = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_stem_loc_y = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_stem_siz = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )

        # iloop
        self.out_iloop_on = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_iloop_loc_x = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_iloop_loc_y = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_iloop_siz_x = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_iloop_siz_y = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )

        # hloop
        self.out_hloop_on = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_hloop_loc_x = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_hloop_loc_y = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )
        self.out_hloop_siz = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(20, 11, kernel_size=1),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc(x)

        # stem
        y_stem_on = self.out_stem_on(x)
        y_stem_loc_x = self.out_stem_loc_x(x)
        y_stem_loc_y = self.out_stem_loc_y(x)
        y_stem_siz = self.out_stem_siz(x)

        # iloop
        y_iloop_on = self.out_iloop_on(x)
        y_iloop_loc_x = self.out_iloop_loc_x(x)
        y_iloop_loc_y = self.out_iloop_loc_y(x)
        y_iloop_siz_x = self.out_iloop_siz_x(x)
        y_iloop_siz_y = self.out_iloop_siz_y(x)

        # hloop
        y_hloop_on = self.out_hloop_on(x)
        y_hloop_loc_x = self.out_hloop_loc_x(x)
        y_hloop_loc_y = self.out_hloop_loc_y(x)
        y_hloop_siz = self.out_hloop_siz(x)

        # collect
        y = {
            # stem
            'stem_on': y_stem_on,
            'stem_location_x': y_stem_loc_x,
            'stem_location_y': y_stem_loc_y,
            'stem_size': y_stem_siz,

            # iloop
            'iloop_on': y_iloop_on,
            'iloop_location_x': y_iloop_loc_x,
            'iloop_location_y': y_iloop_loc_y,
            'iloop_size_x': y_iloop_siz_x,
            'iloop_size_y': y_iloop_siz_y,

            # hloop
            'hloop_on': y_hloop_on,
            'hloop_location_x': y_hloop_loc_x,
            'hloop_location_y': y_hloop_loc_y,
            'hloop_size': y_hloop_siz,
        }

        return y


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
        assert set(x).issubset(set(list('ACGTN')))
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



def _make_mask(l):
    m = np.ones((l, l))
    m[np.tril_indices(l)] = 0
    return m


def predict_bounidng_box(pred_on, pred_loc_x, pred_loc_y, pred_siz_x, pred_siz_y, thres=0.5):
    # remove singleton dimensions
    pred_on = np.squeeze(pred_on)
    pred_loc_x = np.squeeze(pred_loc_x)
    pred_loc_y = np.squeeze(pred_loc_y)
    pred_siz_x = np.squeeze(pred_siz_x)
    if pred_siz_y is None:
        pred_siz_y = np.copy(pred_siz_x)
    # TODO assert on input shape

    # hard-mask
    m = _make_mask(pred_on.shape[1])
    # apply mask (for pred, only apply to pred_on since our processing starts from that array)
    pred_on = pred_on * m
    # binary array with all 0's, we'll set the predicted bounding box region to 1
    # this will be used to calculate 'sensitivity'
    pred_box = np.zeros_like(pred_on)
    # also save box locations and probabilities
    proposed_boxes = []

    for i, j in np.transpose(np.where(pred_on > thres)):
        loc_x = np.argmax(pred_loc_x[:, i, j])
        loc_y = np.argmax(pred_loc_y[:, i, j])
        siz_x = np.argmax(pred_siz_x[:, i, j]) + 1  # size starts at 1 for index=0
        siz_y = np.argmax(pred_siz_y[:, i, j]) + 1
        # compute joint probability of taking the max value
        prob = pred_on[i, j] * softmax(pred_loc_x[:, i, j])[loc_x] * softmax(pred_loc_y[:, i, j])[loc_y] * \
               softmax(pred_siz_x[:, i, j])[siz_x - 1] * softmax(pred_siz_y[:, i, j])[siz_y - 1]  # FIXME multiplying twice for case where y is set to x
        # top right corner
        bb_x = i - loc_x
        bb_y = j + loc_y
        # save box
        proposed_boxes.append({
            'bb_x': bb_x,
            'bb_y': bb_y,
            'siz_x': siz_x,
            'siz_y': siz_y,
            'prob': prob,   # TODO shall we store 4 probabilities separately?
        })
        # set value in pred box, be careful with out of bound index
        x0 = bb_x
        y0 = bb_y - siz_y + 1  # 0-based
        wx = siz_x
        wy = siz_y
        ix0 = max(0, x0)
        iy0 = max(0, y0)
        ix1 = min(x0 + wx, pred_box.shape[0])
        iy1 = min(y0 + wy, pred_box.shape[1])
        pred_box[ix0:ix1, iy0:iy1] = 1

    # apply hard-mask to pred box
    pred_box = pred_box * m
    return proposed_boxes, pred_box



def make_plot_bb(target, pred_box):
    fig = px.imshow(target)
    for bb in pred_box:
        bb_x = bb['bb_x']
        bb_y = bb['bb_y']
        siz_x = bb['siz_x']
        siz_y = bb['siz_y']
        prob = bb['prob']

        x0 = bb_x
        y0 = bb_y - siz_y + 1  # 0-based
        wx = siz_x
        wy = siz_y
        fig.add_shape(
            type='rect',
            y0=x0 - 0.5, y1=x0 + wx - 0.5, x0=y0 - 0.5, x1=y0 + wy - 0.5,  # image plot axis is swaped
            xref='x', yref='y',
            opacity=prob,  # opacity proportional to probability of bounding box
            line_color='red'
        )

    # update figure
    fig['layout'].update(height=800, width=800, title="test")

    return fig


def cleanup_hloop(bbs, l):
    # remove hloop that's not on diagonal
    bbs_new = []
    for bb in bbs:
        bb_x = bb['bb_x']
        bb_y = bb['bb_y']
        siz_x = bb['siz_x']
        siz_y = bb['siz_y']
        y0 = bb_y - siz_y + 1
        x1 = bb_x + siz_x - 1
        if bb_x == y0 and bb_y == x1:
            bbs_new.append(bb)
    return bbs_new


class Predictor(object):

    def __init__(self, model_ckpt):
        # model_ckpt: model params checkpoint
        # FIXME fixed architecture for now, training workflow need to save this in a config
        model = SimpleConvNet(num_filters=[32, 32, 64, 64, 64, 128, 128],
                              filter_width=[9, 9, 9, 9, 9, 9, 9], dropout=0)
        model.load_state_dict(torch.load(model_ckpt, map_location=torch.device('cpu')))
        # TODO print model summary
        self.model = model

    def predict_bb(self, seq, threshold):
        de = DataEncoder(seq)
        yp = self.model(torch.tensor(de.x_torch))
        yp = {k: v.detach().cpu().numpy()[0, :, :, :] for k, v in yp.items()}
        # bb
        pred_bb_stem, _ = predict_bounidng_box(pred_on=yp['stem_on'], pred_loc_x=yp['stem_location_x'],
                                               pred_loc_y=yp['stem_location_y'], pred_siz_x=yp['stem_size'],
                                               pred_siz_y=None, thres=threshold)
        pred_bb_iloop, _ = predict_bounidng_box(pred_on=yp['iloop_on'], pred_loc_x=yp['iloop_location_x'],
                                                pred_loc_y=yp['iloop_location_y'], pred_siz_x=yp['iloop_size_x'],
                                                pred_siz_y=yp['iloop_size_y'], thres=threshold)
        pred_bb_hloop, _ = predict_bounidng_box(pred_on=yp['hloop_on'], pred_loc_x=yp['hloop_location_x'],
                                                pred_loc_y=yp['hloop_location_y'], pred_siz_x=yp['hloop_size'],
                                                pred_siz_y=None, thres=threshold)
        pred_bb_hloop = cleanup_hloop(pred_bb_hloop, len(seq))
        return yp, pred_bb_stem, pred_bb_iloop, pred_bb_hloop


class Evaluator(object):

    def __init__(self, predictor):
        assert isinstance(predictor, Predictor)
        self.predictor = predictor
        # hold on to data for one example, for convenience
        self.data_encoder = None
        # predictions
        self.yp = None
        self.pred_bb_stem = None
        self.pred_bb_iloop = None
        self.pred_bb_hloop = None

    def predict(self, seq, y, threshold):
        assert 0 <= threshold <= 1
        # y: in one_idx format, tuple of two lists of i's and j's
        self.data_encoder = DataEncoder(seq, y, bb_ref='top_right')
        yp, pred_bb_stem, pred_bb_iloop, pred_bb_hloop = self.predictor.predict_bb(seq, threshold)
        self.yp = yp
        self.pred_bb_stem = pred_bb_stem
        self.pred_bb_iloop = pred_bb_iloop
        self.pred_bb_hloop = pred_bb_hloop

    def plot(self):
        # TODO plot titles
        fig_stem = make_plot_bb(self.data_encoder.y_arrs['stem_on'], self.pred_bb_stem)
        fig_iloop = make_plot_bb(self.data_encoder.y_arrs['iloop_on'], self.pred_bb_iloop)
        fig_hloop = make_plot_bb(self.data_encoder.y_arrs['hloop_on'], self.pred_bb_hloop)
        # TODO add more plots
        return fig_stem, fig_iloop, fig_hloop

    # TODO metrics
