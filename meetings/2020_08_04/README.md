## Process dataset

bpRNA (used by SPOT-RNA): successful


Target values:
per-pixel classification

background mask (gradient masking)

one pixel can take on multiple classes!
e.g. a pixel can be both top right corner of hairpin loop and bottom left corner of stem

if we come up with a scheme to assign probability/class to every pixel,
then we can compute the joint probability.
to avoid exponentially many assembly possibilities,
we can apply cut off to construct an initial set of proposals
before running the discrete step.

debug: terminal internal loop? (does not make sense)


## Training local structure

```
python train_nn.py --data data/local_struct.bp_rna.pkl.gz --result result/debug --num_filters 16 32 32 64 --num_stacks 4 4 4 4 --epoch 10 --batch_size 20 --max_length 80 --cpu 4
```


on GPU:
```
CUDA_VISIBLE_DEVICES=0 python train_nn.py --data data/local_struct.bp_rna.pkl.gz --result result/debug --num_filters 16 32 32 64 --num_stacks 4 4 4 4 --epoch 10 --batch_size 20 --max_length 80 --cpu 4
```



TODOs:

- *DONE* remove training data points with non-ACGTN character

- *DONE* loss: average over spatial dimension.
due to masking, we'll be summing and dividing by the number of valid entries.

- export prediction (training + validation) & visualize (make better plot + caption)

- smaller input region? localize to the structure? (same as masking most of the background)

- toy example?

- per-channel naive guess & performance

- sample negative mask (make sure to always mask lower triangle)

- report metric on different outputs

- other ways to parametrize output?

- run on GPU

- discover intrinsic structure within dataset (e.g. families?) unsupervised?

- deep learning assemble proposal?

- formulate as RL?

