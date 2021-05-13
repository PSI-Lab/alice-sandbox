## S1 dataset

- try to improve stem bb sensitivity

- since we're only dealing with stem bb now

- not only predicting stem bb in MFE struct (since we still have potential FN
even when we mask most of the 'background')

- no need to limit MFE struct freq when generating data?

- for each sequence, we'll sample a couple structures using RNAfold (unique?),
and 'merge' their stem bbs

- challenge: stem bbs from different structures might be
in conflict with each other



- test dataset: no merge, since we'll be evaluating s1 bb sensitivity





