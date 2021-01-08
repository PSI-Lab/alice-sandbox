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

```
195000 77.518718957901
Traceback (most recent call last):
  File "model_utils/run_stage_1.py", line 77, in <module>
    main(args.data, args.num, args.threshold, args.topk, args.model, args.out_file)
  File "model_utils/run_stage_1.py", line 53, in main
    uniq_stem, uniq_iloop, uniq_hloop = predictor.predict_bb(seq, threshold, topk)
  File "/home/alice/work/psi-lab-sandbox/meetings/2021_01_05/model_utils/utils_model.py", line 687, in predict_bb
    result = sm_top_one(pred_on[i, j], pred_loc_x, pred_loc_y, pred_sm_siz_x, pred_sm_siz_y, i, j)
  File "/home/alice/work/psi-lab-sandbox/meetings/2021_01_05/model_utils/utils_model.py", line 675, in _predict_bb
    # apply mask (for pred, only apply to pred_on since our processing starts from that array)
  File "/home/alice/work/psi-lab-sandbox/meetings/2021_01_05/model_utils/utils_model.py", line 629, in predict_bounidng_box
    setting cutoff == 1 correspond to picking the argmax"""
  File "/home/alice/work/psi-lab-sandbox/meetings/2021_01_05/model_utils/utils_model.py", line 532, in sm_top_k
    assert top_idx[3] == np.argmax(siz_y)
AssertionError

```

- `13c48ee..5388782` Separate probabilities for softmax/scalar output unit predicted bb.
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

- `5388782..4e8a579` topk & top_perc: for pixel where bb_on > threshold, use bbs whose joint probability is within a certain percentage of the top hit,
AND only use up to k bbs.

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
uniq_stem, uniq_iloop, uniq_hloop = predictor_s1.predict_bb(seq, threshold=0.02, topk=5, perc_cutoff=0.5)
uniq_stem, uniq_iloop, uniq_hloop = predictor_s1.predict_bb(seq, threshold=0.02, topk=20, perc_cutoff=0.5)
```

- `79b12c5..f47492f` -> `f47492f..f89e27e` update run_s1 script and re-run inference, to prepare dataset for s2

sample 5000 for debug training:

```
python model_utils/run_stage_1.py --data "`dcl path ZQi8RT`" --num 5000 --threshold 0.1 --topk 10 --perc_cutoff 0.8 --model v1.0 --out_file data/synthetic_s1_pred_5000.pkl.gz
```


full:


- TODO for scalar size, also save the difference between real valued prediction and the rounded integer?

- TODO: even with topk, the same pixel won't predict k identical bb (softmax/scalar),
so normalizing factor should stay the same? include a warning in data processing.


- Added top k prediction (TODO s2 training data processing update normalization)

- instead of top k, pick all prediction within 90%? of the argmax? run inference
(but we can't determine the normalizing factor of s2 data)

- TODO separate softmax and scalar prediction

- TODO eval: number of bbs old v.s. new, performance old v.s. new

TODO concrete example

TODO debug

```
In [36]: df = pd.read_pickle('../data/synthetic_s1_pred_5000.pkl.gz')

In [37]: len(df)
Out[37]: 4539
```

## S2 training update

### pruning

- `03e19e8..b5deec5` -> `b5deec5..c96250d` -> `c96250d..014b8cd` update to be compatible with new s1 inference output format

debug

```
python model_utils/prune_stage_1.py --in_file data/synthetic_s1_pred_5000.pkl.gz --out_file data/synthetic_s1_5000_pruned.pkl.gz --min_pixel_pred 1 --min_prob 0.1 --min_hloop_size 2 --discard_ns_stem
```

```
[(1506, "'prob_sm'"), (16299, "'prob_sm'"), (80813, "'prob_sm'"), (164275, "'prob_sm'"), (499793, "'prob_sm'"), (151855, "'prob_sm'"), (404563, "'prob_sm'"), (416422, "'prob_sm'"), (258234, "'prob_sm'"), (113048, "'prob_sm'"), (481662, "'prob_sm'"), (105158, "'prob_sm'"), (362755, "'prob_sm'"), (212875, "'prob_sm'"), (426563, "'prob_sm'"), (490598, "'bb_x'"), (399096, "'prob_sm'"), (356428, "'prob_sm'"), (42350, "'prob_sm'"), (331451, "'prob_sm'"), (421457, "'prob_sm'"), (405595, "'prob_sm'"), (346363, "'prob_sm'"), (416941, "'prob_sm'"), (97170, "'prob_sm'")]
```

### Feature encoding


- `f8d1076..9e481c7` now that s1 inference generates two sets of probabilities, update data processing code.
Different feature encoding for softmax/scalar prediction. Now each unique bb has the following features:

    - 1-hot encoding of bb type  (3 features)

    - location and size (4 features)

    - median probability and number of proposals (normalized), from softmax size prediction (2 features)

    - median probability and number of proposals (normalized), from scalar size prediction (2 features)

We also note that even when k > 1 for topk prediction, normalizing factor is not affected,
since if a pixel predicts k bbs, the bbs are different by definition,
thus the max number of pixel that can predict the same bb is unchanged.

In [s2_training/](s2_training/):

debug

```
python make_dataset.py --in_file ../data/synthetic_s1_5000_pruned.pkl.gz  --out_file ../data/synthetic_s2_5000_features.npz
```


- data augmentation

how to deal with missing value? sm/sl

TODO batch mode (otherwise super slow)

TODO re-run S1 model s.t. each pixel predict multiple bbs,
instead of taking argmax of all softmax, sample a few and take the highest k joint probability

synthetic: add in missing bb (how?)


### training

- update config (n_input 9->10)



In [s2_training/](s2_training/):



debug:

```
CUDA_VISIBLE_DEVICES=0 python train_s2.py --in_file ../data/synthetic_s2_5000_features.npz --config config.yml --out_dir result/synthetic_5000/ 2>&1 | tee result/synthetic_5000/log.txt

```

WIP debugging...


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




