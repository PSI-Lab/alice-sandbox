import os
import numpy as np
import pandas as pd
from keras.models import load_model


class UPbmEnsemble(object):

    def __init__(self, tf_name, dataset_name, aggregation, root_dir='result/'):
        # find sequence length
        training_log_file = os.path.join(root_dir, tf_name, '%s.csv' % dataset_name)
        df = pd.read_csv(training_log_file)
        seq_len = int(df[df['task'] == 'training_sequence_length'].iloc[0]['val'])
        print seq_len
        # find model files (hard-coded fold idx for now)
        models = []
        for i in range(5):
            model_file = os.path.join(root_dir, tf_name, '%s.fold_%d.h5' % (dataset_name, i))
            models.append(UPbmModel(model_file, seq_len))
        self.models = models
        # aggregation should be a np operation
        self.aggregation = aggregation

    def predict_sequence(self, seq):
        x = []
        for model in self.models:
            x.append(model.predict_sequence(seq))
        return self.aggregation(np.asarray(x), axis=0)


class UPbmModel(object):
    SEQ_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self, model_file, seq_len):
        print("Loading model %s with training sequence length %d" % (model_file, seq_len))
        self.model = load_model(model_file)
        self.seq_len = seq_len

    def _encode_sequences(self, seqs):
        assert isinstance(seqs, list)
        assert all([isinstance(s, str) for s in seqs])
        assert all([len(s) == len(seqs[0]) for s in seqs])
        data = []
        for seq in seqs:
            assert all([x in 'ACGTNacgtn' for x in seq])
            # TODO make sure sequence is longer than the required length determined by all conv layers
            seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
            x = np.asarray(map(int, list(seq)))
            x = self.SEQ_ENCODING[x.astype('int8')]
            data.append(x)
        return np.swapaxes(np.swapaxes(np.stack(data, axis=2), 0, 2), 1, 2)

    def _predict_sequence(self, seq):
        assert len(seq) == self.seq_len
        data = self._encode_sequences([seq])
        return self.model.predict(data)[0, 0]

    def predict_sequence(self, seq):
        if len(seq) <= self.seq_len:
            data = self._encode_sequences([seq])
            return self.model.predict(data)[0, 0]
        else:
            # sliding window
            list_seqs = []
            for offset in range(len(seq) - self.seq_len + 1):
                list_seqs.append(seq[offset:offset+self.seq_len])
            data = self._encode_sequences(list_seqs)
            return self.model.predict(data)[:, 0]

    def predict_batch_sequences(self, seqs):
        raise NotImplementedError
