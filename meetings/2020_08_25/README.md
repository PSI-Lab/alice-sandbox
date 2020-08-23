## Ideas

### General code improvement

- make memory-efficient dataset:
only store one_idx and bounding box type/location in the dataframe.
Expand into array at training time for each minibatch.

```
python make_dataset_pixel_bb.py
```

Data uploaded to DC: `DmNgdP`

- update training script to use new dataset format

debug:

```
python train_simple_conv_net_pixel_bb.py --data DmNgdP --result result/test_simple_conv_pixel_bb_debug --num_filters 16 --filter_width 9 --epoch 10 --batch_size 20 --max_length 80 --cpu 4
```

GPU debug, 0 worker
```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb.py --data DmNgdP --result result/simple_conv_net_pixel_bb_soft_mask_3 --num_filters 32 32 32 64 64 64 128 --filter_width 9 9 9 9 9 9 9 --epoch 20 --mask 0.1 --batch_size 40 --max_length 200 --cpu 0
```

- GPU run with multiple CPU workers - works now

- generate dataset using synthetic data

```
python make_dataset_synthetic.py
```

Data uploaded to DC: `xs5Soq`


- generate dataset using synthetic data  (bigger dataset)

```
python make_dataset_synthetic_2.py
```

Data uploaded to DC: `ZQi8RT`


### Generate new dataset using synthetic sequence and RNAfold predicted structure

Dataset: `xs5Soq`

```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb.py --data xs5Soq --result result/rf_data_1 --num_filters 32 32 32 64 64 64 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0.1 --batch_size 40 --max_length 200 --cpu 8
```

Observations:

    - lower cross entropy than real dataset: easier to train on?

    - overfitting not as severe as real dataset: probably due to much bigger datset size

    - same as real dataset, even when we overfit (loss on validation increases),
    metrics such as au-roc, au-prc and accuracies stay flat.
    This might be also due to the fact that the loss was computed using soft mask
    (i.e. back ground pixels get count in with a discount factor),
    but the metrics were computed by applying hard mask (background pixels ignored).

Generate plots:

```
python visualize_prediction_pixel_bb.py --in_file result/rf_data_1/pred_ep_10.pkl.gz --out_file result/rf_data_1/plot/ep_10.html --row sample --tgt --bb
python visualize_prediction_pixel_bb.py --in_file result/rf_data_1/pred_ep_49.pkl.gz --out_file result/rf_data_1/plot/ep_49.html --row sample --tgt --bb

python visualize_prediction_pixel_bb.py --in_file result/rf_data_1/pred_ep_10.pkl.gz --out_file result/rf_data_1/plot/ep_10.bb_all_5.html --threshold 0.5 --row all --bb
python visualize_prediction_pixel_bb.py --in_file result/rf_data_1/pred_ep_10.pkl.gz --out_file result/rf_data_1/plot/ep_10.bb_all_4.html --threshold 0.4 --row all --bb
python visualize_prediction_pixel_bb.py --in_file result/rf_data_1/pred_ep_10.pkl.gz --out_file result/rf_data_1/plot/ep_10.bb_all_3.html --threshold 0.3 --row all --bb
python visualize_prediction_pixel_bb.py --in_file result/rf_data_1/pred_ep_10.pkl.gz --out_file result/rf_data_1/plot/ep_10.bb_all_2.html --threshold 0.2 --row all --bb
python visualize_prediction_pixel_bb.py --in_file result/rf_data_1/pred_ep_10.pkl.gz --out_file result/rf_data_1/plot/ep_10.bb_all_1.html --threshold 0.1 --row all --bb
```
(above run on workstation, df not compatible with pandas on laptop?)

Result: https://drive.google.com/drive/folders/13FZheN9AEY-eHssGSe96WQN7a8O5DQUt


### Add output specific hidden layer

```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb.py --data xs5Soq --result result/rf_data_2 --num_filters 32 32 32 64 64 64 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0.1 --batch_size 40 --max_length 200 --cpu 8
```


result: todo (rf_data_2)


### All targets

debug run:

```
python train_simple_conv_net_pixel_bb_all_targets.py --data DmNgdP --result result/debug_2 --num_filters 16 --filter_width 9 --epoch 10 --batch_size 20 --max_length 80 --cpu 4
```


gpu:

```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb_all_targets.py --data xs5Soq --result result/rf_data_all_targets_1 --num_filters 32 32 32 64 64 64 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0.1 --batch_size 40 --max_length 200 --cpu 8
```


plot

```
python visualize_prediction_pixel_bb_all_targets.py --in_file result/rf_data_all_targets_1/pred_ep_20.pkl.gz --out_file result/rf_data_all_targets_1/plot/ep_20.bb_all_5.html --threshold 0.5 --row all --bb --verbose
```


### All targets, larger dataset & model

```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb_all_targets.py --data ZQi8RT --result result/rf_data_all_targets_2 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0.1 --batch_size 40 --max_length 200 --cpu 8
```





## Ideas & TODOs

Make sure to log the conclusion for each idea, for future reference.
(make one section for each idea, move above)
(also for each idea include git hash so we can check the associated training code)

- update pandas on workstation - done

- save model, implement script to predict on new dataset

- enumerate all valid configurations of bounding boxes, how to do it efficiently?

- train on bigger dataset, with more complicated model, see if we can improve sensitivity

- data generator: return extra info, so that we can store at training/validation minibatch.
e.g. sequence, sequence length, original bounding box list

- vectorize bounding box proposal code

- right now we make separate plot for different box types,
 if we plot all on the same, need to use different color for different box type

- the confidence of a bounding box is reflected by:
(1) the joint probability of sigmoid/softmax output,
(2) how many other bounding box overlap

- ideas for assembly:
start with most confidence box,
for each iloop find the compatible stems on each side,
for each hloop find the compatible stem.
formulate as another prediction problem?
given the sequence and all bounding boxes, predict the 'groud truth' combination of bounding box?
equivalent to predict (globally) on/off of each bounding box? how to represent bounding box, another feature map?

- assembly: how to train the 2nd stage algorithm if he first stage sensitivity is not 100%?
i.e. given that we know what is valid path and we have a way to enumerate all valid paths,
how to define closeness of each path to the ground truth? convert it back to binary matrix and calculate the old metrics?

- validity:
stem is not valid if not joined by loop (e.g. hloop between diagonal),
hloop is not valid if it's not across diagonal,
etc.

- bounding box loss (not differentiable?)
TP & FP at threshold, TP: proposed bb overlapping ground truth,
FP: proposed bb not overlapping ground truth.
For the bb module, we want high TP, even at a cost of high FN.

- bounding box size > 10? how?  re-process dataset to partition into smaller (overlapping) boxes?
how to determine which one to assign for each pixel, the closer one?

- if we have the original target bounding box locations, we can compute the following metrics:
(similar to sensitivity):
% of true bounding box that has: identical proposed box, overlapping proposed box, no proposed box.
(similar to specificity):
% of predicted bounding box that has: identical true box, overlapping true box, no overlapping true box

- fix divide warning:

```
  return np.sum(_x2 == _y2)/float(len(_x2))
train_simple_conv_net_pixel_bb.py:579: RuntimeWarning: invalid value encountered in true_divide
  return np.sum(_x2 == _y2)/float(len(_x2))

```

- rnafold generated dataset to pre-train bounding box module (pick example with less ensemble diversity?)

- training (all targets, not just stem)

- incompatibility: stem_on, location shift > 1, but stem size=1

- not able to predict len=1 stem?


- memory efficiency, especially cpu mem (for Linux machine),
sparse array or just save index myself, expand when making minibatch,
use int instead of float

- make minibatch workflow, pass in training=T/F flag to streamline code

- add structured metric logging (for training curve plot)

- improve plot script to compute bb in vectorized format

- add output specific hidden layer

- rectangle -> sequence A & B 'compatibility'?

- long stem -> break into parts (overlapping?), max len 10

- in theory, stem should be really easy to pick up,
investigate alternative architecture? training on patches?

- use sparse array (numpy? scipy?) to improve data loader (taking too much memory)

- read capsule network paper

- train on smaller patches. in theory this should be roughly equivalent to
applying local un-masking, but it's a good baseline to establish anyways.
We can generate patches that center at different types of structures.

- pre-train stage 1 on very short seq, e.g, miRNA or RNAfold generated dataset


- smaller input region? localize to the structure? (same as masking most of the background)
local un-mask

- toy example?

- per-channel naive guess & performance

- if we come up with a scheme to assign probability/class to every pixel,
then we can compute the joint probability.
to avoid exponentially many assembly possibilities,
we can apply cut off to construct an initial set of proposals
before running the discrete step.

- debug: terminal internal loop? (does not make sense)

- discover intrinsic structure within dataset (e.g. families?) unsupervised?

- deep learning assemble proposal?

- formulate as RL?

