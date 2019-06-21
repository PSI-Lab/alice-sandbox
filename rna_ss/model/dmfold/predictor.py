import numpy as np
import keras
from keras import backend as kb
from train_2d import custom_loss


class PredictorDotbracket(object):
    DNA_ENCODING = np.zeros((5, 4))
    DNA_ENCODING[1:, :] = np.eye(4)

    def __init__(self, model_file, context):
        self.model = keras.models.load_model(model_file, custom_objects={"kb": kb})
        self.context = context

    def predict_seq(self, seq):
        # encode input
        seq = 'N' * (self.context / 2) + seq + 'N' * (self.context / 2)
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('U', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        x = x[np.newaxis, :, :]
        yp = self.model.predict(x)[0, :, :]
        out = ''
        for i in range(yp.shape[0]):
            _p = yp[i, :]
            _idx = np.argmax(_p)
            if _idx == 0:
                out += '('
            elif _idx == 1:
                out += '['
            elif _idx == 2:
                out += '{'
            elif _idx == 3:
                out += '.'
            elif _idx == 4:
                out += '}'
            elif _idx == 5:
                out += ']'
            elif _idx == 6:
                out += ')'
            else:
                raise ValueError
        return out


class Predictor2D(object):
    DNA_ENCODING = np.zeros((5, 4))
    DNA_ENCODING[1:, :] = np.eye(4)
    LEN = 500  # hard-coded to match the pre-built graph size
    BS = 10  # batch size is also fixed in training, need to match that -_-

    def __init__(self, model_file):
        self.model = keras.models.load_model(model_file, custom_objects={"kb": kb, "custom_loss": custom_loss})
        # self.context = context

    def predict_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('U', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        _val_h = np.repeat(x[:, np.newaxis, :], len(seq), axis=1)
        _val_v = np.repeat(x[np.newaxis, :, :], len(seq), axis=0)
        _tmp_seq_map = np.concatenate((_val_h, _val_v), axis=2)
        _seq_map = np.zeros((self.LEN, self.LEN, 8))
        _seq_map[:len(seq), :len(seq), :] = _tmp_seq_map
        _seq_map = _seq_map[np.newaxis, :, :, :]
        # need to match batch size unfortunately
        _seq_map = np.repeat(_seq_map, self.BS, axis=0)
        yp = self.model.predict(_seq_map)[0, :, :]
        return yp
