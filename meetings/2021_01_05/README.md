
## Datasets

Sources:

- S1 training data, synthetic sequences: `ZQi8RT`


Intermediate:



Result?? (for plotting?):




## Code organization

### Top level utils

Copied from `../../rna_ss/` as [rna_ss_utils/](rna_ss_utils/).

### S1 training

- Copied from https://github.com/PSI-Lab/alice-sandbox/tree/35b592ffe99d31325ff23a14269cd59fec9d4b53/meetings/2020_11_10#debug-stage-1-training

- Added real valued bb size output, to enable predicting bb with size > 10. Code update: data processing `make_target_pixel_bb`,
training `s1_training/train_simple_conv_net_pixel_bb_all_targets.py`.
and inference `todo`.

- Added missing Relu for 1st layer FC.


- No scaling down on MSE loss (to match dynamic range) since it's quite straight forward for the optimizer (from empirical observation).

- Update plot training progress code:

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
python model_utils/plot_training.py --in_log s1_training/result/with_scalar_size/run.log  --out_plot s1_training/result/with_scalar_size/training_progress.html
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




## Read paper

### DeepSets




### Set Transformer


### DeepSetNet: Predicting Sets with Deep Neural Networks

### Joint Learning of Set Cardinality and State Distribution

### BRUNO: A Deep Recurrent Model for Exchangeable Data


### Deep Set Prediction Networks




## TODOs

- dataset: '../2020_11_24/data/rfam151_s1_pruned.pkl.gz'  'data/synthetic_s1_pruned.pkl.gz'

- inference pipeline: deal with cases where some types of bb are empty

- inference pipeline debug + improvement: n_proposal_norm > 1, implementation using queue, terminate condition

- s2 training: stems only? how to pass in background info like sequence? memory network? encoding?

- s2 training: add batch mode (debug to make sure it works), save model, set up inference utils so we can run the model

- s2 inference: greedy sampling with hard constraints (white & black list)

- s2 training dataset, for those example where s1 bb sensitivity < 100%, add in the ground truth bbs for contructing dataset for s2.
How to set features like median_prob and n_proposal_norm? Average in the same example?

- rfam151 (and other dataset): evaluate base pair sensitivity and specificity (allow off by 1?)

- evaluate sensitivity if we allow +/-1 shift/expand of each bb

- if above works and we have a NN for stage 2, we can feed in this extended set of bb proposals!

- attention -> output set?

- stage 1 prevent overfitting (note that theoretical upper bound is not 100% due to the way we constructed the predictive problem)

- upload best model to DC?

- evaluate rfam stage 2 predictions, majority are not identical, but are they close enough?

- investigate pseudoknot predictions, synthetic dataset (45886-32008)

- try running RNAfold and allow C-U and U-U (and other) base pairs, can we recover the lower FE structure that our model predicts?

- rfam151 dataset debug, is the ground truth bounding box correct? (make sure there’s no off-by-1 error)

- stage 1 model: iloop size = 0 on my side is bulge, make sure we have those cases!

- RNAfold performance on rfam151

- to debug: index 0 with length 117 and n_bbs 21 seems to be stuck during parsing.: python model_utils/run_stage_2.py --in_file data/rfam151_s1_bb_0p1.pkl.gz --out_file data/debug.pkl.gz --min_pixel_pred 3 --min_prob 0.5

- to debug: rfam151, RF00165_A, global structure contain invalid ones (implied iloop and hloop not included):
```
   bb_x  bb_y  siz_x  siz_y bb_type  n_proposal  prob_median  n_proposal_norm
0     1    17      2      2    stem           4     0.137343              1.0
1     4    45      8      8    stem          64     0.859667              1.0
2    27    57     10     10    stem         100     0.721043              1.0
.((.((((((((....)).........((((((((((.))))))))..)))))))))).... 14.362449399658637 100007.1
```

- stage 2, pick the first bb by sampling from all bb's (proportional to the 'likelihood' of the bb?),
then the next ones are picked by some attention based NN? black list & white list?

- extra constraints in stages 2? stem box needs to satisfy G-C, A-U, G-U base pairing (discard those that are not),
min hloop size?

- table documenting all DC IDs (datasets, models, etc.)


- Heuristics: More structure is better -> if global struct A is subset of B, discard A

- Pseudo knot?

- RNA-RNA interaction? Run stage 1 model three times, A-A, B-B & A-B, 2nd stage will have different constraints

- Long sequence?

- Greedy approach of assembly? Start with high prob bounding boxes, terminate after explored say 100 global structures?

- size > 10


