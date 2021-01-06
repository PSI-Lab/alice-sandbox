Continued from last week:

## Datasets

`6PvUty`: rnastralign?

`903rfx`: rfam151?

`a16nRG`: s_processed?

`xs5Soq`: synthetic?

`ZQi8RT`: synthetic? with prediction?

bpRNA?

Sources:

- S1 training data, synthetic sequences: `ZQi8RT`


Intermediate:



Result?? (for plotting?):


## For testing utils interface

TODO update parameters

```python
import model_utils.utils_model as us1
import model_utils.utils_s2 as us2 # TODO merge s2 util
from model_utils.utils_nn_s2 import predict_wrapper

predictor_s1 = us1.Predictor('v1.0')
predictor_s2 = us2.Predictor('s2_training/result/synthetic/model_ckpt_ep_28.pth')

discard_ns_stem = True
min_hloop_size = 2
# these paramters should be consistent with how we trained the s2 model
topk = 1
m_factor = 2 * topk # compatible w/ topk=1

seq = 'ACGATGACGATAGACGCGACGACAGCGAT'

uniq_stem, uniq_iloop, uniq_hloop = predictor_s1.predict_bb(seq, threshold=0.02, topk=topk)
df_pred = predict_wrapper(uniq_stem, uniq_iloop, uniq_hloop, discard_ns_stem=True, min_hloop_size=2, seq=seq, m_factor=m_factor, predictor=predictor_s2)
```


## S1 inference update

check in  env.yml


- WIP running inference with top-2 prediction

- Separate probabilities for softmax/scalar output unit predicted bb.
For scalar predicted bb, only include joint probability of the two location softmax.
(no longer need to be 'compatible' with the softmax-size output since they won't get aggregated in s2 feature!)

```
   bb_x  bb_y  siz_x  siz_y                                            prob_sm                                            prob_sl
0     1    20      2      2  [0.13323026951860742, 0.15145190820696794, 0.1...  [0.13414366017942744, 0.1532410054271312, 0.11...
1     1    26      2      2         [0.03666791894895221, 0.05569421539945699]         [0.0372739099538199, 0.056445935979234134]
2     4    17      2      2  [0.05753037817173854, 0.052009115090564356, 0....  [0.05847623896762833, 0.05243706219068787, 0.0...
3     7    15      2      2  [0.17595531032127867, 0.1839970953597975, 0.05...  [0.17786414010863807, 0.18647292494855539, 0.0...
4    14    26      3      3  [0.8683679106068021, 0.8838014862709586, 0.854...  [0.8708090196355063, 0.8861603020200729, 0.856...
```

- topk & top_perc: for pixel where bb_on > threshold, use bbs whose joint probability is within a certain percentage of the top hit,
only use up to k bbs.  Code updated TODO link

Test:

```
import model_utils.utils_model as us1
predictor_s1 = us1.Predictor('v1.0')
seq = 'ACGATGACGATAGACGCGACGACAGCGAT'

# only topk
uniq_stem, uniq_iloop, uniq_hloop = predictor_s1.predict_bb(seq, threshold=0.02, topk=1)
uniq_stem, uniq_iloop, uniq_hloop = predictor_s1.predict_bb(seq, threshold=0.02, topk=2)

# only top_perc
uniq_stem, uniq_iloop, uniq_hloop = predictor_s1.predict_bb(seq, threshold=0.02, topk=0, perc_cutoff=0.9)

# both
uniq_stem, uniq_iloop, uniq_hloop = predictor_s1.predict_bb(seq, threshold=0.02, topk=5, perc_cutoff=0.9)

# encoding? (add to df? do not merge softmax and scalar)
```



- Added top k prediction (TODO s2 training data processing update normalization)

- instead of top k, pick all prediction within 90%? of the argmax? run inference
(but we can't determine the normalizing factor of s2 data)

- TODO separate softmax and scalar prediction

- TODO eval: number of bbs old v.s. new, performance old v.s. new

TODO concrete example

## S2 training update

### Feature encoding

different feature encoding for softmax/scalar prediction

how to deal with missing value? sm/sl

TODO batch mode (otherwise super slow)

TODO re-run S1 model s.t. each pixel predict multiple bbs,
instead of taking argmax of all softmax, sample a few and take the highest k joint probability

synthetic: add in missing bb (how?)

## S1 evaluation

Would they be recovered if using top k or top percent inference (instead of argmax)?
TODO



## Read paper

### DeepSets




### Set Transformer


### DeepSetNet: Predicting Sets with Deep Neural Networks

### Joint Learning of Set Cardinality and State Distribution

### BRUNO: A Deep Recurrent Model for Exchangeable Data


### Deep Set Prediction Networks




## TODOs

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




