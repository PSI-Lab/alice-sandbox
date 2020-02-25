

## Debug

Shuffle target value (for training not for testing), and inspect the performance:

https://github.com/PSI-Lab/alice-sandbox/tree/9ed0d6968cbef4d647489341604d5fd71cb741c5/meetings/2020_02_28/yeast_model_training

```
python train_base_line_nn.py --data ~/work/psi-lab-sandbox/meetings/2020_02_11/d_cell/data/training/raw_pair_wise/SGA_ExE.txt --result result/exe_shuffle_target --epoch 20 --shuffle_label
```

run log: https://github.com/PSI-Lab/alice-sandbox/blob/5243edffa09d8daf85a39bb57b597b5be2014859/meetings/2020_02_28/yeast_model_training/result/exe_shuffle_target/run.log


Without shuffling (we already ran it last time, but re-running it here anyways):

```
python train_base_line_nn.py --data ~/work/psi-lab-sandbox/meetings/2020_02_11/d_cell/data/training/raw_pair_wise/SGA_ExE.txt --result result/exe_no_shuffle --epoch 20
```

run log: https://github.com/PSI-Lab/alice-sandbox/blob/5243edffa09d8daf85a39bb57b597b5be2014859/meetings/2020_02_28/yeast_model_training/result/exe_no_shuffle/run.log

Comparing the two run logs, and notice the following differences:

- After shuffling target values for the training set,
performance drop to that similar of a naive solution
(which makes sense, since the only gradient direction that's consistent across all minibatches,
 should be pointing towards the naive solution),
and it's consistent between training and testing


TODO read their paper again: how did process the data? were they not trying to predict the raw values?
-> didn't find anythung useful


target value distribution

try on a bigger dataset

visualize network?

anyone has time to help with some code review?

train/test on single deletion

TODO: make GPU version code

where is the bigger (8M) dataset? ref 15


