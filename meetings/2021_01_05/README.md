
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




## Code organization

### Top level utils

Copied from `../../rna_ss/` as [rna_ss_utils/](rna_ss_utils/).

### S1 training

- Copied from https://github.com/PSI-Lab/alice-sandbox/tree/35b592ffe99d31325ff23a14269cd59fec9d4b53/meetings/2020_11_10#debug-stage-1-training

- Added real valued bb size output, to enable predicting bb with size > 10. Code update: data processing `make_target_pixel_bb`,
training `s1_training/train_simple_conv_net_pixel_bb_all_targets.py`.

- Added missing Relu for 1st layer FC.


- No scaling down on MSE loss (to match dynamic range) since it's quite straight forward for the optimizer (from empirical observation).

- Update plot training progress code: `model_utils/plot_training.py`, to include metric on scalar valued target.

- TODO check in training env.yml

See [s1_training/](s1_training/).


Run inside [s1_training/](s1_training/):


Training (todo update output dir, hyperparam?):

```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb_all_targets.py --data ZQi8RT --result result/with_scalar_size --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0.1 --batch_size 40 --max_length 200 --cpu 12
```

(observed slight overfitting, killed at ep 35)

plot training progress:

```
python model_utils/plot_training.py --in_log s1_training/result/with_scalar_size/run.log  --out_plot s1_training/result/with_scalar_size/training_progress.html
```

TODO upload trained model.

```
# train on random sequence, after adding in scalar target for bb size, ep 11
# produced by: https://github.com/PSI-Lab/alice-sandbox/tree/f8df78da280b2a3ba16960a6226afaef2facd734/meetings/2021_01_05#s1-training
'v1.0': 'KOE6Jb',
```



### S1 inference

- update model def `SimpleConvNet` to match training code

- Update inference code to use scalar valued bb size prediction, in addition to softmax.

- How to get joint prob (needed for S2) of bb? set to 1 (might bias S2)? use local Gaussian approximation since we need to round (how to set the std?)?
For now, we assume the predictive distribution is Gaussian with mean y0 and std 1, the predicted size is y (y0 rounded to int),
and we use the ratio: `pdf(y)/pdf(y0)`, which in the standardized form of Gaussian(0, 1): `norm.pdf(y-y0)/norm.pdf(0)`


- scalar output: avoid setting size 0 or negative, set those to 1: `x[x < 1] = 1`

- Added new version of trained S1 model `'v1.0': 'KOE6Jb'`

- Added top k prediction (TODO s2 training data processing update normalization)

- WIP running inference with top-2 prediction



- TODO eval: number of bbs old v.s. new, performance old v.s. new

- instead of top k, pick all prediction within 90%? of the argmax? run inference
(but we can't determine the normalizing factor of s2 data)

```
uniq_stem, uniq_iloop, uniq_hloop = predictor_s1.predict_bb(seq, threshold=0.02, topk=0, perc_cutoff=0.9)
```

TODO concrete example

- Run inference pipeline to produce dataset for S2 training:

Synthetic:

argmax (top 1):

```
python model_utils/run_stage_1.py --data "`dcl path ZQi8RT`" --num 0 --threshold 0.1 --model v1.0 --out_file data/synthetic_s1_pred.pkl.gz
```

top k where k = 2 (running):

```
python model_utils/run_stage_1.py --data "`dcl path ZQi8RT`" --num 0 --threshold 0.1 --topk 2 --model v1.0 --out_file data/synthetic_s1_pred.topk2.pkl.gz
```

sample 50000 for debug training:

```
python model_utils/run_stage_1.py --data "`dcl path ZQi8RT`" --num 50000 --threshold 0.1 --model v1.0 --out_file data/synthetic_s1_pred_50000.pkl.gz
```

rnastralign (TODO):

```
python model_utils/run_stage_1.py --data "`dcl path 6PvUty`" --num 0 --threshold 0.1 --model v1.0 --out_file data/rnastralign_s1_pred.pkl.gz
```




Rfam (TODO long se OOO on Linux? need more efficient inference pipeline):


```
python model_utils/run_stage_1.py --data "`dcl path 903rfx`" --num 0 --threshold 0.1 --model v1.0 --out_file data/rfam_s1_pred.pkl.gz
```



#### Using the model


```
from model_utils.utils_model import Predictor
predictor = Predictor('v1.0')
predictor.predict_bb('ACGTGTACGATGCAG', 0.1)
```


- run on multiple datasets: synthetic, rfam, etc.


- interface for predicting RNA-RNA interaction w/ different lengths

### S1 evaluation


on synthetic dataset:


```
python model_utils/eval_model_dataset.py --data "`dcl path xs5Soq`" --num 200 --maxl 200 --model v1.0 --out_csv result/synthetic_s1_debug.csv --out_plot result/synthetic_s1_debug.html
```


bounding box metric

non-100% bb sensitivity: how about shift/expand bb?
how about using not just argmax of softmax?

pixel metric

focus on sensitivity

any improvement after adding in scalar output?


### S2 training

#### Prune S1 predicted bounding boxes for S2 training

synthetic with relaxed thresholds:

```
python model_utils/prune_stage_1.py --in_file data/synthetic_s1_pred.pkl.gz --out_file data/synthetic_s1_pruned.pkl.gz --min_pixel_pred 1 --min_prob 0.1 --min_hloop_size 2 --discard_ns_stem 2>&1 | tee data/log_synthetic_s1_pruned.txt
```

debug

```
python model_utils/prune_stage_1.py --in_file data/synthetic_s1_pred_50000.pkl.gz --out_file data/synthetic_s1_50000_pruned.pkl.gz --min_pixel_pred 3 --min_prob 0.5 --min_hloop_size 2 --discard_ns_stem 2>&1 | tee data/log_synthetic_s1_50000_pruned.txt
```

#### Feature generation


synthetic:

In [s2_training/](s2_training/):

```
python make_dataset.py --in_file ../data/synthetic_s1_pruned.pkl.gz  --out_file ../data/synthetic_s2_features.npz
```

####  training

In [s2_training/](s2_training/):


<!--```-->
<!--CUDA_VISIBLE_DEVICES=0 python train_s2.py --in_file ../data/synthetic_s1_pruned.pkl.gz --config config.yml --out_dir result/synthetic/ 2>&1 | tee result/synthetic/log.txt-->

<!--```-->

<!--debug-->


<!--```-->
<!--CUDA_VISIBLE_DEVICES=0 python train_s2.py --in_file ../data/synthetic_s1_50000_pruned.pkl.gz --config config.yml --out_dir result/debug/ 2>&1 | tee result/debug/log.txt-->
<!--```-->


synthetic:

```
CUDA_VISIBLE_DEVICES=0 python train_s2.py --in_file ../data/synthetic_s2_features.npz --config config.yml --out_dir result/synthetic/ 2>&1 | tee result/synthetic/log.txt

```


update: n_proposed_normalized: denominator * 2 since each pixel can predict the same bb twice now (softmax and scalar size)

Copied from TODO

TODO batch mode (otherwise super slow)

TODO re-run S1 model s.t. each pixel predict multiple bbs,
instead of taking argmax of all softmax, sample a few and take the highest k joint probability

TODO run S1 model

TODO data augmentation: add bb location shift (should be invariant to small global shift)
also, what about large shift? 'empty space'?

encoding: shall we distinguish bb predict by softmax or scalar? no?

synthetic: add in missing bb

training on rfam? add in missing bb?


TODO re-train using more bb's from S1

upstream:

downstream:

### Inference

s1 + s2:

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



