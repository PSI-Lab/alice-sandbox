
## Paper

The "human transcriptome" dataset is essentially synthetic dataset
with input sequence being slides of human genome (as opposed to uniformly random bases).


## Re-consider S1

Thoughts:

- use human transcriptome

- since in S1, we're only interested in predicting 2D local bounding box,
sequence does not need to be very long (need to be at least larger than the receptive field)

- add boundary cropping (due to conv receptive field) <- not directly (we're need to deal with boundary at inference time).
we should pad input sequence with N's, then crop the prediction.

- train on only those examples where MFE is the mode and account for majority of the prob mass? (less uncertainty
in terms of local bounding box?) or sample from ensemble and use high probability bbs (are there any?)

- seq_len: conv filter size? proportion of pixels with real context v.s. not?

- is conv the right architecture?
are we predicting a smooth function?

- re-examine NN architecture

- redundancy in 1st layer conv filter weights: input is redundant, only need the 'center line' both horizontally and vertically

### Generate new training and testing dataset for S1

- human genes, random non-overlapping slices of certain length

- length picked to be sufficient for receptive field (so at least some pixels get
all real bases in the full receptive field) (in the early runs
I was using 50 which is a bit small (mis-calculated), corrected in "run 6")

- require MFE strcuture frequency to be > 10%

- use chr1 for testing, and the rest for training/validation

debug

```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 50 --num_seq 10 --threshold_mfe_freq 0.1 --chromosomes chr1 --out ../data/debug.pkl.gz
```

training set 1000:

```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 50 --num_seq 1000 --threshold_mfe_freq 0.1 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_1000.pkl.gz
```

testing set 100:


```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 50 --num_seq 100 --threshold_mfe_freq 0.1 --chromosomes chr1 --out ../data/human_transcriptome_segment_high_mfe_freq_testing_100.pkl.gz
```




### Training

i- n theory: no need to pad since conv net padding should be all zeros,
but this is not true (remember the case I debugged last time when writing patch-prediction pipeline?)

- this is because: 0-padding at each conv layer is not identical to pad 0's for the total context length in the first layer.
In the latter, subsequent layers gets the value of `bias` instead of 0.


- fixed architecture: we can explicit pad when calling conv1d for the first layer,
and do not use padding for any subsequent layers

-verify maskingL default unmaking local offset 10 applies only to on/off units,
for other units, pixels beyond the bounding box
are not penalized, i.e. all masked (since we don't know the true target)

<!--equivariant & invariant? pixel shift -> bb rel loc shift but size fixed-->

<!--try: let it overfit?-->



"run 1"

GPU

soft mask turned off (note we should set cmd line arg mask = 0, since this is the value to be set for hard_mask=0.
setting to 0 basically means use original hard mask) <- only affecting on/off TODO link to code, more epochs

```
cd s1_training/
CUDA_VISIBLE_DEVICES=0 python train_conv_pixel_bb_fixed_length_padding.py --data ../data/human_transcriptome_segment_high_mfe_freq_training_1000.pkl.gz --result result/run_1 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 200 --mask 0 --batch_size 20 --max_length 200 --cpu 8
```

(overfit? - expected since dataset is small)


### Generate more training examples

training set 10000:

```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 50 --num_seq 10000 --threshold_mfe_freq 0.1 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_10000.pkl.gz
```


### Training

"run 2"

with 10k training examples

```
cd s1_training/
CUDA_VISIBLE_DEVICES=0 python train_conv_pixel_bb_fixed_length_padding.py --data ../data/human_transcriptome_segment_high_mfe_freq_training_10000.pkl.gz --result result/run_2 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 200 --mask 0 --batch_size 20 --max_length 200 --cpu 8
```


### Generate more training examples


training set 50000:

```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 50 --num_seq 50000 --threshold_mfe_freq 0.1 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_50000.pkl.gz
```

### Training


"run 3"

with 50k training examples

```
cd s1_training/
CUDA_VISIBLE_DEVICES=1 python train_conv_pixel_bb_fixed_length_padding.py --data ../data/human_transcriptome_segment_high_mfe_freq_training_50000.pkl.gz --result result/run_3 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 200 --mask 0 --batch_size 20 --max_length 200 --cpu 8
```


### Generate more training examples


training set 100000:

```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 50 --num_seq 100000 --threshold_mfe_freq 0.1 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_100000.pkl.gz
```

### Training


"run 4"

with 100k training examples

```
cd s1_training/
CUDA_VISIBLE_DEVICES=0 python train_conv_pixel_bb_fixed_length_padding.py --data ../data/human_transcriptome_segment_high_mfe_freq_training_100000.pkl.gz --result result/run_4 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 200 --mask 0 --batch_size 20 --max_length 200 --cpu 8
```


(this one seems to be good enough??)

killed at epoch 154


### Generate more training examples


training set 200000:

```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 50 --num_seq 200000 --threshold_mfe_freq 0.1 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_200000.pkl.gz
```


### Training


"run 5"

with 200k training examples

```
cd s1_training/
CUDA_VISIBLE_DEVICES=1 python train_conv_pixel_bb_fixed_length_padding.py --data ../data/human_transcriptome_segment_high_mfe_freq_training_200000.pkl.gz --result result/run_5 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 200 --mask 0 --batch_size 20 --max_length 200 --cpu 8
```


## Re-train S1 after Saturday meeting

- increase sequence length to 64 (original 50 was a bit under the size of receptive field, which
was 28 on each side)

- use L1 loss on scalar units, so we get better precision

### Generate training and test dataset

debug set (for local testing) len=60, size=100:

```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 64 --num_seq 100 --threshold_mfe_freq 0.1 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_debug_len64_100.pkl.gz
```

training set len=64, size=200000:

```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 64 --num_seq 200000 --threshold_mfe_freq 0.1 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_len64_200000.pkl.gz
```

testing set len=64, size=100:


```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 64 --num_seq 100 --threshold_mfe_freq 0.1 --chromosomes chr1 --out ../data/human_transcriptome_segment_high_mfe_freq_testing_len64_100.pkl.gz
```


### Training

- update scalar unit loss

- add loss and metric logging csv file, for easy visualization

code update: `0f509e1..ed6be1a`


"debug run"

```
cd s1_training/
python train_conv_pixel_bb_fixed_length_padding.py --data ../data/human_transcriptome_segment_high_mfe_freq_debug_len64_100.pkl.gz --result result/debug --num_filters 4 4  --filter_width 9 9 --epoch 2 --mask 0 --batch_size 20 --max_length 200 --cpu 0
```

"run 6", 50 epochs (no need to run 200 according to previous experience)


```
cd s1_training/
CUDA_VISIBLE_DEVICES=0 python train_conv_pixel_bb_fixed_length_padding.py --data ../data/human_transcriptome_segment_high_mfe_freq_training_len64_200000.pkl.gz --result result/run_6 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0 --batch_size 20 --max_length 200 --cpu 8
```


running



Sanity check training progress (partial): https://docs.google.com/presentation/d/1qzjzpHrnJLYIsANx7zqU2a8Gj8XbAInSUOFR1zB5EZY/edit#slide=id.gc8469e9b72_0_0
See tmp_s1_plot_train_progress_csv.ipynb




### Eval

- TODO clean up S1 inference code

- TODO Update S1 inference to discard prediction based on the last unit of softmax (those capturing e.g. >10)

- TODO Check S1 bounding box sensitivity on training dataset (as well as test set)

- TODO Validate a few more examples of different bounding boxes (see https://vectorinstitute.slack.com/archives/DAE7HH8QH/p1615672851006200)




code update:  `git_hash`





## Re-analyze multi-branch loop on 2D grid


## S2

sweep, look up S1 pred, RL?



Investigate architecture - TODO


### Eval


#### Eval on same length test set

len=50


#### Longer seq

Generate test set of len 100:


test set len=100, 100 examples:


```
cd s1_data_gen
python generate_human_transcriptome_segment_high_mfe_freq.py --len 100 --num_seq 100 --threshold_mfe_freq 0.1 --chromosomes chr1 --out ../data/human_transcriptome_segment_high_mfe_freq_testing_len_100_100.pkl.gz
```


#### Other dataset?


## Re-consider S2

Thoughts:

- how to pass in original sequence?

- multi-branch loop?

- DP on bounding boxes? how to train?


## Datasets

(be careful: old dataset using top left corner format)

`6PvUty`: rnastralign?

`903rfx`: rfam151?

`a16nRG`: s_processed?

`xs5Soq`: synthetic?

`ZQi8RT`: synthetic? with prediction?

`DmNgdP`: bpRNA?

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



