

## Bounding box assembly

See WIP report [2020_10_06.pdf](2020_10_06.pdf)


## Pointer net

cont'd form last week


is accuracy only coming from the first output?


## Hard-wired stem-only bounding box

cont'd form last week

analyze sensitivity

other dataset?

## bounding box assembly with constraints

Prediction from stage 1 model:




## New todos

- minimal plot utils

- pointer network to mimic linearfold?

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
