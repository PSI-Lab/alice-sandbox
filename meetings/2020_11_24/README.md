
[model_utils/](/model_utils) are copied from last week's work.
Refer to notes from `2020_11_10` on how to use it as a package.

## Stage 2 update

Incorporate in hard constraints for pruning bounding box proposal.
Add as optional parameters.

- discard stems that contain non standard base pairs (standard: A-U, G-C, G-U)

- discard hairpin loops that have size <= 2

For example, this is valid:

```
GUCCCCC
UAGGGGG
```

while this is invalid:

```
GUCCAGUC
GUAGGUCG
```


re-run stage 2 on rfam:

```
python model_utils/run_stage_2.py --in_file data/rfam151_s1_bb_0p1.pkl.gz --out_file data/rfam151_s2_3_0p5.pkl.gz --max_len 100 --min_pixel_pred 3 --min_prob 0.5 --min_hloop_size 2 --discard_ns_stem 2>&1 | tee data/log_rfam151_s2_3_0p5.txt
```


Use RNAfold to compute FE for all predicted structures:

```
python compute_s2_fe.py --in_file data/rfam151_s2_3_0p5.pkl.gz --out_file_fe data/rfam151_s2_3_0p5_fe.pkl.gz --out_file_score data/rfam151_s2_3_0p5_all_scores.pkl.gz
```

Eval FE:

TODO


Eval base pair sensitivity & specificity:

TODO


## Read paper

### DeepSets


### Set Transformer


### DeepSetNet: Predicting Sets with Deep Neural Networks

### ﻿Joint Learning of Set Cardinality and State Distribution

### ﻿BRUNO: A Deep Recurrent Model for Exchangeable Data


### Deep Set Prediction Networks




## TODOs

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
