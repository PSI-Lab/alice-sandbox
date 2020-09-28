
## Pointer net

### Generate input stream and features

Focus on synthetic dataset since it's bb performance is so far the best,
as a proof of concept for 2nd stage model.

Script: [../2020_09_22/generate_bb_target_idx.py](../2020_09_22/generate_bb_target_idx.py)

Result: [data/rand_s1_bb_0p1_features.pkl.gz](data/rand_s1_bb_0p1_features.pkl.gz)   (renamed from `tmp.pkl.gz` which was moved from last week)

Example:

```
In [6]: df.iloc[0].features.shape
Out[6]: (29, 13)

In [7]: df.iloc[0].target_idx
Out[7]: [0, 2, 5, 10, 27, 1]

In [8]: df.iloc[0].bb_overlap
Out[8]: [1, 1.0, 1.0, 1.0, 1.0, 1]

In [10]: df.iloc[0].input_len
Out[10]: 29
```

Each example has the following:

- `features`: np array of features for each bounding box (each will be one 'timestamp' in the encoder RNN),
where features include bounding box type (1-hot encoding, with the first 2 entries reserved to placeholder timestamp `start` and `end`),
closing base pair positions, summary statistics of stage 1 probabilities.

- `target_idx`: this is the sequential indices the PointerNet needs to predict.

- `input_len`: redundant, stored for convenience. Number of bounding boxes proposed by stage 1 model,
this is also the number of timestamps in PointerNet encoder.

- `bb_overlap`: overlap with ground truth for each target bounding box (output index).
First and last entry are always `1.0` since they are `start` and `end` placeholders.


Caveats:

- Haven't finalized bounding box ordering, might cause conflicting gradient while training NN.


### Pointer net - org

Implementation adapted from https://github.com/ast0414/pointer-networks-pytorch/tree/27aec4e011a37f96270e3d6ac7423225d4c40ef3

Some small improvements:

- code reformating: tab replaced by 4 spaces

- fix deprecation warning: use bool for masked_fill()

Test the original implementation on sorting example:

```
python train_sort.py --low 2 --high 10 --min-length 2 --max-length 5 --train-samples 1000 --test-samples 20 --emb-dim 20 --batch-size 20 --no-cuda --epochs 20
```

Seems to be working:

```
Epoch 19: Train [0/1000 (0%)]	Loss: 0.392546	Accuracy: 0.848495
Epoch 19: Train [400/1000 (40%)]	Loss: 0.386630	Accuracy: 0.851058
Epoch 19: Train [800/1000 (80%)]	Loss: 0.380774	Accuracy: 0.853624
Epoch 19: Test	Loss: 0.401322	Accuracy: 0.860000
```


### Data Generator update

Update data generator to use our dataset.

For this experiment, we'll be focusing on examples where
stage 1 bounding box sensitivity is `100%`.

[pointer_net/dataset_ss.py](pointer_net/dataset_ss.py)


### Model update

[pointer_net/model_ss.py](pointer_net/model_ss.py)

- removed linear embedding layer since it's causing NaN gradient (not sure why?)

### Training update

[pointer_net/train_ss.py](pointer_net/train_ss.py)

### Training

Sanity check:

```
python train_ss.py --dataset ../data/rand_s1_bb_0p1_features.pkl.gz --num-layers 1 --hid-dim 20 --batch-size 20 --no-cuda --epochs 2 --wd 0
```

Run on GPU:

```
CUDA_VISIBLE_DEVICES=0 python train_ss.py --dataset ../data/rand_s1_bb_0p1_features.pkl.gz --num-layers 3 --hid-dim 50 --batch-size 200 --epochs 100
```



## bb assembly with constraints



## Hard-wired stem-only bb



## New todos

- 'feature' of a proposed stem can be summarized to fixed dimension by an RNN

- cubic time for finding all stem bbs

- bpRNA summary: percentage of examples wiht non-canonical base_pairing, percentage of paired bases with non canonical pairing

- stage 1 output valid assemblies: how?

- stage 1 only predict(or hard-wired) stem bb's (localy max size), stage 2 use pointer net, how to feed in background sequence information?

- linearfold? pointer net?

- Vectorize bounding box proposal code

- implement (efficient) hard-wired logic to find all (locally maximum) stem boxes: stage 1 training data, stage 2 input data

- how to propagate background/global information (sequence?)

- upload a few existing models to DC

- bpRNA local structure size histogram

- process bpRNA 1M(90) dataset, also compare my code with their local structures, any mistakes in my code? (why is iloop_loc_y performance always 100%?)

- try to control overfit on bpRNA, regularization?

- bounding box assembly <-> shotgun sequencing? (although we know the location)

- add real value output for loc & size, so we can predict on those with size > 10

- process bpRNA original dataset, use 1M(90)? (with 90% similarity threshold) (be careful to not overfit!)

- try stage 2 ideas in parallel, we can use model trained on synthetic dataset since its bb performance is best

- dynamic batch size depend on max length?

- training script: instead of saving all predictions,
save the predicted boudning boxes at threshold 0.1 (move inference code to top level?)

- stem bb (hard-wired logic, complexity?) + pointer net?

- debug inference & eval code

- plot: rand model performance on rand dataset

- plot: number of local structures per sequence length, rand v.s. bpRNA

- try training on other dataset: RNAstralign?

- experiment: use bpRNA seq, use RNAfold to predict structure, then train & test on this dataset,
is the performance better now?

- dropout at inference time

- predict on bpENA dataset (test set, never seen before), eval bb performance

- NN combinatorics

- RL combinatorics

- bounding box indices: memory address, nucleotide: memory content?  RNN?

- ensemble? bb union of v0.1 and v0.2 model

- stage 2: hloop, given top right corner, the size is deterministic, we can discard those whose predicted size does not match

## Ideas & TODOs

Make sure to log the conclusion for each idea, for future reference.
(make one section for each idea, move above)
(also for each idea include git hash so we can check the associated training code)

- update pandas on workstation - done

- save model, implement script to predict on new dataset - done

- now we're saving metadata, update plot code cropping since we no longer need to infer length,
also add evaluation to check identical/overlapping box

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

