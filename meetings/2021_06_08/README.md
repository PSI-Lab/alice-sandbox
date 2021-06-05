Last week:

- Direct application of GNN to solve S2 problem seem to have hit a bottleneck


## Is max bps a good heuristic for global free enerygy?

### Generate dataset


- for now focus on short seq with fewer bb proposals (before we implement efficient search)

- generate fixed length short seq, len=40, no threshold on mfe_freq:

env `rna_ss_py3`

```
cd s2_data_gen
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py \
--len_min 40 --len_max 40 --num_seq 1000 --threshold_mfe_freq 0 \
--chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 \
--out ../data/human_transcriptome_segment_high_mfe_freq_training_len40_1000.pkl.gz
```


- run S1 inference:

debug

```
python run_s1_inference_run32_ep79.py --data ../2021_06_01/data/s2_train_len20_200_2000_pred_stem_0p5.pkl.gz \
--threshold 0.5 --out data/debug.pkl.gz --num 10
```

env `pytorch_plot_py3`

```
taskset --cpu-list 21,22,23,24 python run_s1_inference_run32_ep79.py --data data/human_transcriptome_segment_high_mfe_freq_training_len40_1000.pkl.gz \
--threshold 0.5 --out data/data_len40_1000_s1_stem_bb.pkl.gz
```

- generate all valid stem bb combinations

- for now, subset to those with bb sensitivity = 100%,
subset to those with <= 15 s1 proposal bbs (since we're brute-forcing the search)

```
python process_dataset_for_scoring_network_training.py
```

output: `data/data_len40_1000_s1_stem_bb_le10_combos.pkl.gz`


### Compare MFE with best from top 10

- for each valid stem bb combination, using `RNAeval` to compute its free energy

- some combos have FE > 0 (pseudoknot? I thought RNAeval returns a super large number? no? did they update?)
(see example ID 18 max bp struct)


![plot/mfe_vs_top_10.png](plot/mfe_vs_top_10.png)

- x axis: FE of the MFE structure by RNAfold

- y axis: min FE of the top-10 stem bb combinations, ranked by number of bps

- note that to achieve this performance, we'll need:

    1. a perfect S2 scoring model that can rank structure in the same order as FE

    2. a perfect search procedure that can give us top-k

### Visualize a few examples

![plot/rank_by_bps_examples.png](plot/rank_by_bps_examples.png)


Produced by [num_bps_vs_fe.ipynb](num_bps_vs_fe.ipynb)





## S2 scoring network


- training in a siamease setting with tied weights

### Dataset




- loss function? do not penalize if p(x1) > p(x0)?


## S2 binary tree search with constraints

- implementation

- heuristic: use S1 probability?


## Alternative formulation of the combinarorial problem

- valid combination of bbs

- convert to a known CO?


## DP formulation on bb?

- can we decompose global score? if so,
we can use DP to significantly cut down the computation.
No need to score every single valid bb combination separately.

- Can we do it in a way to still allow for pseudoknot?

- can we train it? RL?






## Read papers
