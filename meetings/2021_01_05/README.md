
## Datasets

Sources:

- S1 training data, synthetic sequences: `ZQi8RT`


Intermediate:



Result?? (for plotting?):




## Code organization

### Top level utils

Copied from `../../rna_ss/` as [rna_ss_utils/](rna_ss_utils/).

### S1 training

Copied from https://github.com/PSI-Lab/alice-sandbox/tree/35b592ffe99d31325ff23a14269cd59fec9d4b53/meetings/2020_11_10#debug-stage-1-training

Added real valued bb size output, to enable predicting bb with size > 10.
Code update: data processing `make_target_pixel_bb`,
training `s1_training/train_simple_conv_net_pixel_bb_all_targets.py`
and inference `todo`.

Added missing Relu for 1st layer FC.


TODO scale down MSE loss? (dynamic range)

See [s1_training/](s1_training/).


Run inside [s1_training/](s1_training/):

debug:

```
python train_simple_conv_net_pixel_bb_all_targets.py --data ZQi8RT --result result/debug --num_filters 16 16 --filter_width 9 9 --epoch 2 --mask 0.1 --batch_size 10 --max_length 40 --cpu 1
```


Training (todo update output dir, hyperparam?):

```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb_all_targets.py --data ZQi8RT --result result/with_scalar_size --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0.1 --batch_size 40 --max_length 200 --cpu 12
```


plot training progress:

```
python model_utils/plot_training.py --in_log result/with_scalar_size/run.log  --out_plot result/with_scalar_size/training_progress.html
```

TODO upload trained model.

upstream:

downstream:

### S2 training

Copied from TODO

TODO run S1 model

TODO re-train using more bb's from S1

upstream:

downstream:

### Inference

Copied from [../2020_12_15/](../2020_12_15/), renamed to TODO

upstream:


## Evaluation

### Run S1 model

TODO bb cutoff?

### Run S2 model






