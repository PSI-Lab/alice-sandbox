import numpy as np
import keras
import logging
import genome_kit as gk
from model import resolve_contex


class DataGenerator(keras.utils.Sequence):

    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self, config, chroms):
        self.genome = gk.Genome(config['genome_annotation'])
        self.train_length = config['example_length']
        self.context = resolve_contex(config['dense_conv'])
        self.batch_size = config['batch_size']

        self.intervals = self._get_intervals(chroms)
        # shuffle indexes before generating data
        self.on_epoch_end()

    def _get_intervals(self, chroms):
        """list of sub intervals to train on"""
        intervals = []
        n_skipped = 0
        for gene in self.genome.genes:
            if gene.chromosome in chroms:
                itv = gene.interval
                # make sure there's no NaN in data
                data = self.genome.rnaplfold_unpair_prob(itv)
                if not np.all(~np.isnan(data)):
                    logging.info("gene length {} nan's {}".format(len(itv), np.sum(np.isnan(data))))
                    if np.sum(np.isnan(data))/float(len(itv)) > 0.5:
                        n_skipped += 1
                        continue

                x = itv.end5.expand(0, self.train_length)
                while x.within(itv):
                    # skip if most data is missing
                    data = self.genome.rnaplfold_unpair_prob(x)

                    intervals.append(x)
                    x = x.shift(self.train_length)

        logging.warning("Total intervals: %d, skipped genes: %d" % (len(intervals), n_skipped))
        return intervals

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.intervals) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        x, y = self.__data_generation(indexes)
        # note that this need to match the model, see model.py
        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.intervals))
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        x, y = self.get_data([self.intervals[i] for i in indexes])

        return x, y

    def get_data_single(self, itv):
        seq = self.genome.dna(itv)
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = self.DNA_ENCODING[x.astype('int8')]
        y = self.genome.rnaplfold_unpair_prob(itv)[:, :]  # second dimension is 1
        # replace nan with -1
        y[np.where(np.isnan(y))] = -1
        return x, y

    def get_data(self, intervals):
        x = []
        y = []
        for itv in intervals:
            _x, _y = self.get_data_single(itv)
            x.append(_x)
            y.append(_y)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y
