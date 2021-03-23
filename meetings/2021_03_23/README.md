
## Re-training S1 model


- fix random seed when spliting training/validation

- use 95% for training and 5% for validation

code update:  `2ff9631..0d41c34`


"run 7"


```
cd s1_training/
CUDA_VISIBLE_DEVICES=0 python train_conv_pixel_bb_fixed_length_padding.py --data ../../2021_03_16/data/human_transcriptome_segment_high_mfe_freq_training_len64_200000.pkl.gz --result result/run_7 --num_filters 32 32 64 64 64 128 128 --filter_width 9 9 9 9 9 9 9 --epoch 50 --mask 0 --batch_size 20 --max_length 200 --cpu 8
```


### Train progress


```
scp alice@alice-new.dg:~/work/psi-lab-sandbox/meetings/2021_03_23/s1_training/result/run_7/metrics.csv s1_training/result/run_7/.
```

Plot metrics: https://docs.google.com/presentation/d/11gBQ9tI6OHQZXewigetl0985Xp-DgUD_U2xWhy68nsg/edit#slide=id.gc8f7586164_0_83

Notes:

    - Metrics going up: au-ROC and au-PRC, going down: absolute difference (on scalar outputs)


    - Validation metrics plateaus after a certain point, while training metrics keeps improving.
    Interestingly we don't see the typical "overfitting" since we don't
    see a drop in validation performance(on metrics at least).


Pick 'best model', by summing up all metrics (abs_diff converted to negative_abs_diff),
and find the best on on validation set. Epoch `17` was picked.

   stem_on.auroc |   stem_on.auprc |   stem_location_x.accuracy |   stem_location_y.accuracy |   stem_sm_size.accuracy |   stem_sl_size.diff |   iloop_on.auroc |   iloop_on.auprc |   iloop_location_x.accuracy |   iloop_location_y.accuracy |   iloop_sm_size_x.accuracy |   iloop_sm_size_y.accuracy |   iloop_sl_size_x.diff |   iloop_sl_size_y.diff |   hloop_on.auroc |   hloop_on.auprc |   hloop_location_x.accuracy |   hloop_location_y.accuracy |   hloop_sm_size.accuracy |   hloop_sl_size.diff | tv         |   epoch |
----------------:|----------------:|---------------------------:|---------------------------:|------------------------:|--------------------:|-----------------:|-----------------:|----------------------------:|----------------------------:|---------------------------:|---------------------------:|-----------------------:|-----------------------:|-----------------:|-----------------:|----------------------------:|----------------------------:|-------------------------:|---------------------:|:-----------|--------:|
        0.989931 |        0.876432 |                   0.976055 |                   0.975939 |                0.966644 |            0.113115 |         0.990416 |         0.821331 |                    0.983674 |                    0.983816 |                   0.967324 |                   0.966693 |               0.189189 |               0.202064 |         0.990274 |         0.845299 |                    0.964678 |                    0.965083 |                 0.972998 |             0.331314 | training   |      17 |
        0.988068 |        0.861711 |                   0.961073 |                   0.960523 |                0.947276 |            0.13826  |         0.987231 |         0.794417 |                    0.954774 |                    0.954977 |                   0.910889 |                   0.91377  |               0.303958 |               0.326201 |         0.988885 |         0.831771 |                    0.934234 |                    0.936293 |                 0.94391  |             0.458773 | validation |      17 |

See [tmp_s1_plot_train_progress_csv.ipynb](tmp_s1_plot_train_progress_csv.ipynb).



### Visualize softmax activation pattern

Plots: https://docs.google.com/presentation/d/11gBQ9tI6OHQZXewigetl0985Xp-DgUD_U2xWhy68nsg/edit#slide=id.gc8f7586164_0_91

Notes:

    - We don't have see the perfect pattern of "pixels within bounding box has sharp activation but those outside do not".
    In fact, some pixels outside are still consistently pointing to the same bounding box location,
    and their argmax size is also the same (not shown on plot). (these are also the reason why sometimes
    n_proposal_norm can be larger than 1).

    - Example was randomly picked, location specified by eye-balling the on/off probability pattern,
    hard to do this type of analysis systematically at scale.

See [tmp_s1_pixel_pred_shift_example.ipynb](tmp_s1_pixel_pred_shift_example.ipynb)





## Subsample training set for evaluation

Subsample 100 examples. (use same rand seed when we split training/validation):

```

In [1]: import pandas as pd

In [2]: df = pd.read_pickle('~/work/psi-lab-sandbox/meetings/2021_03_16/data/human_transcriptome_segment_high_mfe_freq_training_len64_200000.pkl.gz')

In [3]: df = df.sample(frac=1, random_state=5555).reset_index(drop=True)

In [4]: tr_prop = 0.95

In [5]: _n_tr = int(len(df) * tr_prop)

In [6]: df_tr = df[:_n_tr]

In [7]: df_tr_100 = df_tr[:100]

In [8]: len(df_tr_100)
Out[8]: 100

In [9]: df_tr_100.to_pickle('~/work/psi-lab-sandbox/meetings/2021_03_16/data/human_transcriptome_segment_high_mfe_freq_training_len64_rand5555_100.pkl.gz', compressi
   ...: on='gzip')


```


```
scp alice@alice-new.dg:~/work/psi-lab-sandbox/meetings/2021_03_16/data/human_transcriptome_segment_high_mfe_freq_training_len64_rand5555_100.pkl.gz ../2021_03_16/data/.
```



## S1 inference

- Update S1 inference to discard prediction based on the last unit of softmax (those capturing e.g. >10)


## Eval bb sensitivity on training and testing set

See https://docs.google.com/presentation/d/11gBQ9tI6OHQZXewigetl0985Xp-DgUD_U2xWhy68nsg/edit#slide=id.gc8f7586164_0_0

## Investigate missed bounding boxes

why?

check prediction of enclosing pixels

todo add result

## S1 inference new method

ignore on/off

use oc & siz argmax, look for consistent (e.g. num_proposal_norm > 0.5? be careful with hloop) ones?
rank?

threshold on both n_proposal and prob???

no threshold just ranking?

## S2 training

### Generate dataset

- Generate new dataset: variable lengths 20-200, less trigent MFE freq cutoff (since we'll be predicting glocal structure this time)

debug

```
cd s2_data_gen
python generate_human_transcriptome_segment_high_mfe_freq_var_len.py --len_min 20 --len_max 200 --num_seq 100 --threshold_mfe_freq 0.02 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/debug_training_len20_200_100.pkl.gz
```

Dataset 5000:

```
cd s2_data_gen
python generate_human_transcriptome_segment_high_mfe_freq_var_len.py --len_min 20 --len_max 200 --num_seq 5000 --threshold_mfe_freq 0.02 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000.pkl.gz
```

- Run S1 inference: threshold on n_proposal, 0.2

debug

```
python model_utils/run_s1_threshold_on_n_proposal.py --data data/debug_training_len20_200_100.pkl.gz --threshold 0.2 --model s1_training/result/run_7/model_ckpt_ep_17.pth --out_file data/debug_s1_pred_len20_200_100.pkl.gz
```

Dataset 5000:

```
python model_utils/run_s1_threshold_on_n_proposal.py --data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000.pkl.gz --threshold 0.2 --model s1_training/result/run_7/model_ckpt_ep_17.pth --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000.pkl.gz
```

- Prune bounding boxes, remove invalid ones, min hloop size 3, set other to 0 (no pruning)

debug

```
python model_utils/prune_stage_1.py --in_file data/debug_s1_pred_len20_200_100.pkl.gz --discard_ns_stem --min_hloop_size 3 --min_pixel_pred 0 --min_prob 0 --out_file data/debug_s1_pred_len20_200_100_pruned.pkl.gz
```

Dataset 5000:


```
python model_utils/prune_stage_1.py --in_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000.pkl.gz --discard_ns_stem --min_hloop_size 3 --min_pixel_pred 0 --min_prob 0 --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned.pkl.gz
```

- examples with sensitivity<100% discarded

debug

```
python s2_training/make_dataset.py --in_file data/debug_s1_pred_len20_200_100_pruned.pkl.gz --out_file data/debug_s1_pred_len20_200_100_pruned_filtered.npz
```

```
Subset to examples with 100% S1 bb sensitivity (for now). Before 99, after 39
```

Dataset 5000:

```
python s2_training/make_dataset.py --in_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned.pkl.gz --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_filtered.npz
```

```
Subset to examples with 100% S1 bb sensitivity (for now). Before 4949, after 1876
```

Note:

    - compare to the old inference method this seems to yield a much lower number of examples? why? (to be investigated)


- TODO check distribution, n bbs


### Model training

Using old method. TODO reference

debug

```
cd s2_training/
python train_s2.py --in_file ../data/debug_s1_pred_len20_200_100_pruned_filtered.npz --config debug_config.yml --out_dir result/debug/
```

Dataset 5000:

```
cd s2_training/
CUDA_VISIBLE_DEVICES=1 python train_s2.py --in_file ../data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_filtered.npz --config config.yml --out_dir result/run_1/
```


### Encode sequence feature

- update S1 pred pruning to remove bb out of range: `bd0d027..5bc2a03`

```
python model_utils/prune_stage_1.py --in_file data/debug_s1_pred_len20_200_100.pkl.gz --discard_ns_stem --min_hloop_size 3 --min_pixel_pred 0 --min_prob 0 --out_file data/debug_s1_pred_len20_200_100_pruned.pkl.gz
```

- model ideas: S2 with LSTM as sequence encoder. Jamboard: https://jamboard.google.com/d/1exf2uP4ZERQDsTx2A2zEugMcZ8F-vGjMue2KsnuCyC8/viewer?f=0

- Note that in the above encoding,
there's actually intrisic symmetry (e.g. flip for stem),
which could potentially be modeled in a systematic fashion in NN.

- Working on data processing for training, see [tmp_s2_feature_new.ipynb](tmp_s2_feature_new.ipynb)






### S2 with kmer features

Hard to come up with reasonable kmer features. different bb types.


### RL?

As a sanity check, use existing bb features

immediate reward: blackbox

discounted over time: whitebox

final reward: validity, FE (if no pseudoknot? otherwise remove pseudoknot and approximate)

NN: simple version, self attention with bb features and already-predicted labels, output is a big softmax


complex version with background sequence?  attend to sequence?



## Saturday meeting

### Action items from last meeting

[Alice] clean up S1 data generation and inference code, to go through with Andrew in next meeting
[Alice-done] Re-train S1 model with: (1) longer sequence (at least covers the receptive field), (2) use L1 loss on scalar units
[Alice-done] Update S1 inference to discard prediction based on the last unit of softmax (those capturing e.g. >10)
[Alice-done] Check S1 bounding box sensitivity on training dataset
[Alice-done] Validate a few more examples of different bounding boxes (see point 2 above)
[Alice] After S1 model re-trained, train S2 model using the old architecture
[Andrew-done] set up shared folder on Vector cluster

### Multi-branch loop plot

on 2D: always 1 'outer' stem and n-1 'inner' stems,
not a 2D localized bounding box, but is essentially a relationship
between multiple 'closing base pairs'.

(done)



### S1 inference updated

"run 7"

### Pixel-shift plots

(done)

### Walk through S1 inference

(done)

### S1 eval metric bb sensitivity

training set and test set

(done)


### Missing bb & S1 inference by thresholding n_proposal

(done)

### Set up S1 tuning on Vector cluster

### S2 feature, S2 data, S2 training

problem: implied bounding box, e.g. just a stem -> could imply a hloop without explicitly predicting it (not self-consistent)

### S2 ideas and discussion

- update code to use pytorch trnasformer library (keep it as-in)

- generate new training dataset of various lengths 20-200

- S1 inference: threshold on n_proposal, 0.2

- examples with <100% will be diacrsded (until kmer)

- LSTM? or heuristics? k-mers?

- push to new repo (S1 training)

- scp the dataset


TODO check n_proposal v.s. metric (S1)
