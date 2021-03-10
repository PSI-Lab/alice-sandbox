
## S1 inference on long sequence

### array method

Completed last week.

Usage:


```
import model_utils.utils_model as us1
predictor_s1 = us1.Predictor('v1.0')

seq = 'ACGATGACGATAGACGCGTATTAGACGAGACGGACGTAGACGACGACAGCGATGACGATGACGATAGACGACGACAGCGA'

# non-split (original interface)
stem_1, iloop_1, hloop_1 = predictor_s1.predict_bb(seq, threshold=0.1, topk=1, perc_cutoff=0)

# long seq interface
stem_2, iloop_2, hloop_2 = predictor_s1.predict_bb_split(seq, threshold=0.1, topk=1, perc_cutoff=0,
                                                         patch_size=100)

# above should yield exactly same result
```


### Running on datasets

re-generated using threshold=0.1

run on bpRNA?

```
python model_utils/run_stage_1.py --data "`dcl path DmNgdP`" --num 0 --threshold 0.1 --topk 1 --perc_cutoff 0 --patch_size 150 --model v1.0 --out_file data/bprna_t0p1_k1.pkl.gz
```

running


run on s_processed


```
python model_utils/run_stage_1.py --data "`dcl path a16nRG`" --num 0 --threshold 0.1 --topk 1 --perc_cutoff 0 --patch_size 150 --model v1.0 --out_file data/s_processed_t0p1_k1.pkl.gz
```

todo


rnastralign

```
python model_utils/run_stage_1.py --data "`dcl path 6PvUty`" --num 0 --threshold 0.1 --topk 1 --perc_cutoff 0 --patch_size 150 --model v1.0 --out_file data/rnastralign_t0p1_k1.pkl.gz
```

todo

## S1+S2 prediction

### bpRNA?


Sampled 100 examples for visualization.

Plots: https://docs.google.com/presentation/d/1Jkh0eplqOPGmqvcjoEtk0GV8hKzf94hNMuIp1XPeEqM/edit#slide=id.gbd3dfc4e94_0_6

Produced by [s2_eval_bprna.ipynb](s2_eval_bprna.ipynb)

Looks like 'identical bounding box' sensitivity is too low for us to fine tune a S2 model,
since we won't be able to construct the training dataset.

todo: remove examples that don't satisfy hard constraints?
(triple pairing ones should have been already removed since we need to find bounding box,
what about non standard stem base pairing? check pre-processing script)


## S1 training

bpRNA

in [s1_training/](s1_training/):

```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb_all_targets.py --data DmNgdP --result result/with_scalar_size --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0.1 --batch_size 20 --max_length 200 --cpu 8
```

running

## S1 variational inference

local: a distribution of bounding boxes, conditional on a latent variable

### bpRNA

in [s1_training/](s1_training/):

```
CUDA_VISIBLE_DEVICES=1 python train_variational_pixel_bb.py --data DmNgdP --result result/var_1 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --latent_dim 20 --epoch 50 --mask 0.1 --batch_size 20 --max_length 200 --cpu 8
```


### synthetic

```
CUDA_VISIBLE_DEVICES=1 python train_variational_pixel_bb.py --data xs5Soq --result result/var_2 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --latent_dim 20 --epoch 50 --mask 0.1 --batch_size 20 --max_length 200 --cpu 8
```

small dataset for debug? todo


## S2 inference

WIP

s2 inference: sampling (instead of argmax at every step)
s2 inference: topn at each step?
s2 inference: maintain max size stack? hard since recursion depth is unknown


## Read paper

### DeepSets

### DeepSetNet: Predicting Sets with Deep Neural Networks

### Joint Learning of Set Cardinality and State Distribution

### BRUNO: A Deep Recurrent Model for Exchangeable Data


### Deep Set Prediction Networks

## Datasets

(be careful: old dataset using top left corner format)

`6PvUty`: rnastralign?

`903rfx`: rfam151?

`a16nRG`: s_processed?

`xs5Soq`: synthetic?

`ZQi8RT`: synthetic? with prediction?

`DmNgdP`: bpRNA?

Sources:

- S1 training data, synthetic sequences: `ZQi8RT`


Intermediate:


## TODOs

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



