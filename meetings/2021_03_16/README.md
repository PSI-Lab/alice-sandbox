
## CVAE



## Re-consider S1

Thoughts:

- use human transcriptome

- since in S1, we're only interested in predicting 2D local bounding box,
sequence does not need to be very long (need to be at least larger than the receptive field)

- add boundary cropping (due to conv receptive field) <- not directly (we're need to deal with boundary at inference time).
we should pad input sequence with N's, then crop the prediction.

- train on only those examples where MFE is the mode and account for majority of the prob mass? (less uncertainty
in terms of local bounding box?)

- is conv the right architecture?
are we predicting a smooth function?

- re-examine NN architecture

- redundancy in 1st layer conv filter weights: input is redundant, only need the 'center line' both horizontally and vertically

### Generate new training and testing dataset for S1


### Training

debug

```

```


GPU

```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb_all_targets.py --data DmNgdP --result result/with_scalar_size --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0.1 --batch_size 20 --max_length 200 --cpu 8
```

## Re-consider S2

Thoughts:

- how to pass in original sequence?

- multi-branch loop?

- DP on bounding boxes? how to train?



## Paper

###﻿NEURAL REPRESENTATION AND GENERATION FOR RNA SECONDARY STRUCTURES



## TODOs

- train on shorter sequences (be careful with cnn receptive field size), stem only, self attention?

- masked conv (first layer?)

- KL annealing during training

- bounding box -> segmentation map? (with overlapping pixel), rectangular shape?

- CVAE for global prediction

- local struct free energy as objective?

- add in prior-based loss

- small toy example?

- VAE to generate base-pair graph directly? no need for bb? constraints?

- check convergence
check inference
global/local/semi-local z?
training: same z for same bb, overlapping pixel pick random one?
inference: different z for different pixel

- s2 inference: sampling mode, instead of taking argmax at each step (including the starting bb), sample w.r.t. the model output probability

- latent variable model

- when do we predict 'no structure'?

- try a few more params for S1 comparison plot: (1) t=0.02, k=1,c=0, (2) t=0.1,k=0,c=0.9, (3) t=0.1,k=0,c=0.5, ….etc.
generate another random test dataset (use new data format with top right corner)
try t=0.000001
try t=0.000001 and k=2


- s2 idea: stacked 2D map: seq + binary, one for each local structure (predicted by s1). self attn across 2d maps?

- s2 idea: GNN? 'meta' node connects local structure? predict on/off of meta node? still can't incoportate non-local structure

- dataset: '../2020_11_24/data/rfam151_s1_pruned.pkl.gz'  'data/synthetic_s1_pruned.pkl.gz'

- inference pipeline debug + improvement: n_proposal_norm > 1, implementation using queue, terminate condition

- s2 training: stems only? how to pass in background info like sequence? memory network? encoding?

- s2 training dataset, for those example where s1 bb sensitivity < 100%, add in the ground truth bbs for contructing dataset for s2.
How to set features like median_prob and n_proposal_norm? Average in the same example?

- rfam151 (and other dataset): evaluate base pair sensitivity and specificity (allow off by 1?)

- evaluate sensitivity if we allow +/-1 shift/expand of each bb

- if above works and we have a NN for stage 2, we can feed in this extended set of bb proposals!

- stage 1 prevent overfitting (note that theoretical upper bound is not 100% due to the way we constructed the predictive problem)

- investigate pseudoknot predictions, synthetic dataset (45886-32008)

- try running RNAfold and allow C-U and U-U (and other) base pairs, can we recover the lower FE structure that our model predicts?

- rfam151 dataset debug, is the ground truth bounding box correct? (make sure there’s no off-by-1 error)

- stage 1 model: iloop size = 0 on my side is bulge, make sure we have those cases!

- RNAfold performance on rfam151

- RNA-RNA interaction? Run stage 1 model three times, A-A, B-B & A-B, 2nd stage will have different constraints





old dataset in top left corner format, convert everything to top right?



