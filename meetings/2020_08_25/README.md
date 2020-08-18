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

todo: try GPU with more workers

### Generate new dataset using synthetic sequence and RNAfold predicted structure


## Ideas & TODOs

Make sure to log the conclusion for each idea, for future reference.
(make one section for each idea, move above)
(also for each idea include git hash so we can check the associated training code)

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

