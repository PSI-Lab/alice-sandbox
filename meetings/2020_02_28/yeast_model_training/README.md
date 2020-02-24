

## Debug

Shuffle target value (for training not for testing), and inspect the performance:

https://github.com/PSI-Lab/alice-sandbox/tree/4782f7007bba6d9a92ac1b15d054ba1192e1e153/meetings/2020_02_28/yeast_model_training

```
python train_base_line_nn.py --data ~/work/psi-lab-sandbox/meetings/2020_02_11/d_cell/data/training/raw_pair_wise/SGA_ExE.txt --result result/exe_shuffle_target --epoch 20 --shuffle_label
```


Without shuffling (we already ran it last time, but re-running it here anyways):

```
python train_base_line_nn.py --data ~/work/psi-lab-sandbox/meetings/2020_02_11/d_cell/data/training/raw_pair_wise/SGA_ExE.txt --result result/exe_no_shuffle --epoch 20
```

performance drop

loss is low? suspicious?

still gained some performance in training set

try on a bigger dataset

TODO: make GPU version code
