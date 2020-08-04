## Process dataset

- bpRNA (used by SPOT-RNA): successful


- Target values: per-pixel classification.
5 independent sigmoid:
not_local_structure, stem, internal_loop (include bulges), hairpin loop, is_corner.
Independent since multiple units can be 1 at the same time, e.g.
one pixel can be both 'stem' and 'is_corner'.
Also, 'is_corner' denote the intersection between two local structures
(and we can not say it's the corner of which structure since it belongs to both).
See https://docs.google.com/document/d/1rCN1egH31xzDO4YjC7Xhs25y3zHz2gQR9YirD8Y_UU0/edit


- background mask (gradient masking).
Hard-mask: lower triangular & length padding.
local-unmask: rationale:
two local sequences form a structure (partially) because
it's more likely to be structure than other sequences in
the same 2D neighbourhood.
See https://docs.google.com/document/d/1rCN1egH31xzDO4YjC7Xhs25y3zHz2gQR9YirD8Y_UU0/edit





## Training local structure

```
python train_nn.py --data data/local_struct.bp_rna.pkl.gz --result result/debug --num_filters 16 32 32 64 --num_stacks 4 4 4 4 --epoch 10 --batch_size 20 --max_length 80 --cpu 4
```


on GPU:
```
CUDA_VISIBLE_DEVICES=0 python train_nn.py --data data/local_struct.bp_rna.pkl.gz --result result/debug --num_filters 16 32 32 64 --num_stacks 4 4 4 4 --epoch 10 --batch_size 20 --max_length 80 --cpu 4
```


Current observations:

- hard to predict sharp bounding boxes, hard to predict sparse output e.g. is_corner

- overfit?

See https://docs.google.com/document/d/1rCN1egH31xzDO4YjC7Xhs25y3zHz2gQR9YirD8Y_UU0/edit

TODOs:

- *DONE* remove training data points with non-ACGTN character

- *DONE* loss: average over spatial dimension.
due to masking, we'll be summing and dividing by the number of valid entries.

- *DONE* export prediction (training + validation) & visualize (make better plot + caption)

- smaller input region? localize to the structure? (same as masking most of the background)
local un-mask

- toy example?

- per-channel naive guess & performance

- sample negative mask (make sure to always mask lower triangle)

- report metric on different outputs

- other ways to parametrize output?

- run on GPU/Vector cluster

- if we come up with a scheme to assign probability/class to every pixel,
then we can compute the joint probability.
to avoid exponentially many assembly possibilities,
we can apply cut off to construct an initial set of proposals
before running the discrete step.

- debug: terminal internal loop? (does not make sense)

- discover intrinsic structure within dataset (e.g. families?) unsupervised?

- deep learning assemble proposal?

- formulate as RL?