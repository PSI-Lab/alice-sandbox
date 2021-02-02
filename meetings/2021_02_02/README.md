## S2 training

Continued from last week.


### run on full dataset

re-generating data

(backup before pruning)

? (need pruning) (also note the params used by s1 inference, make sure to reflect when running inference <- shall we save it in the wrapper as a known version?)


## S1 inference on long sequence


1. determine trim size based on conv filter sizes.
We don't have dilation, and all filter width are odd number, so this is quite straight-forward.
trim_size on each side is: sum_i(layer_i_filter_width//2).
For example, 3 layers of filter with width of 9 will yield trim_size = 12,
i.e. extend 12 bases on each side of the sub sequence.


2. split seq-seq on 2D grid into patches, each patch has a input region and output region,
where input region correspond to the two sub sequences which will be encoded and fed into the NN,
and output region correspond to the pixels from which we get bounding box predictions from.
Pixels fall outside of the output region is masked at prediction time, i.e. they do not
contribute to bounding box prediction (since these pixel don't 'see' sufficient context)

![plot/long_seq_split.png](plot/long_seq_split.png)

3. run predictor on each patch with output mask, and 'translate' the patch-predicted bounding boxes
by adding in the patch input coordinate of top left corner.

4. after all patches are run, merge bounding box by their coordinate and size.
This is necessary since bounding box sitting across patches could be predicted by more than one patch.


Note that we're doing this rather complicated procedure (as opposed to just predict on patches,
save the prediction matrix and concatenate, then run bounding box prediction on the concatenated array),
since for long sequences, even storing the prediction matrix might be costly.


Sanity check:

- trivial case: seq_len=200, patch_size=200, trim_size=100 (working)

- easy case (no padding): seq_len=200, patch_size=100, trim_size=100 (working, TODO git commit notebook)

- ?: seq_len=56, patch_size=28, trim_size=28 (working! this confirms that trim_size is indeed 28 as expected, but there's something wrong with our padding in wrapper)

- ?: seq_len=56, patch_size=28, trim_size=30 (working, as expected)

- ?: seq_len=56, patch_size=28, trim_size=20 (not working, as expected)

- ?: seq_len=80, patch_size=40, trim_size=28 (not working?)

- ?: seq_len=80, patch_size=40, trim_size=40 (working, this is where each ext_patch is exactly the original arr...)

- ?: seq_len=56, patch_size=30, trim_size=28 (not working?)

- ?: seq_len=112, patch_size=56, trim_size=28 (not working?)

- ?: seq_len=112, patch_size=28, trim_size=28 (not working? so the problem is with middle patches where both sides are extended???)

- ?: seq_len=200, patch_size=100, trim_size=50 (does not work, to debug)

major bug somewhere? is it only working when the ext_patch is the full arr?

TODO tmp work-around: pad input seq to int multiple of 28??

TODO figure out context size -> N padding doesn't seems to be working properly, could also be due to this.
ervert back to 28 after fixing N padding.


TODO clean up code and put in model_utils

TODO run on rfam


 wrapper, split and merge, be careful with trimming conv context

TODO debug to make sure it yield same result

TODO test on rfam

## S2 inference


s2 inference: sampling (instead of argmax at every step)
s2 inference: topn at each step?
s2 inference: maintain max size stack? hard since recursion depth is unknown

## S2 eval

visualize attention weight matrix.

![plot/attn_softmax.png](plot/attn_softmax.png)

Produced by [visualize_attn.ipynb](visualize_attn.ipynb). (rely on a hacked version of predictor class that prints attn softmax, commit `3ffdac2..daa996c`)


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

bpRNA?

Sources:

- S1 training data, synthetic sequences: `ZQi8RT`


Intermediate:


## TODOs

- visualize attention matrix

- s2 inference: sampling mode, instead of taking argmax at each step (including the starting bb), sample w.r.t. the model output probability

- latent variable model

- when do we predict 'no structure'?

- try a few more params for S1 comparison plot: (1) t=0.02, k=1,c=0, (2) t=0.1,k=0,c=0.9, (3) t=0.1,k=0,c=0.5, ….etc.
generate another random test dataset (use new data format with top right corner)
try t=0.000001
try t=0.000001 and k=2

- s1 inference: running on longer sequence, can create a wrapper of the existing interface:
seq -> short seq pairs -> dfs -> translate -> stitch -> prediction. Be careful with boundary effect.
bb across boundary > include all.

- s2 idea: stacked 2D map: seq + binary, one for each local structure (predicted by s1). self attn across 2d maps?

- s2 idea: GNN? 'meta' node connects local structure? predict on/off of meta node? still can't incoportate non-local structure

- S1 inference pipeline for super long sequences: break into chunks and stitch

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






old dataset in top left corner format, convert everything to top right?




