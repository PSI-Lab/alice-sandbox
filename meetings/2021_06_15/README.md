Last week:

We've demonstrated feasibility of 3 things, using a small dataset with len=40 sequences:

- all valid stem bb combinations can be recovered by binary tree search with constraints

- a scoring network for global structure can be learned via siamese network training

- num_bps is a reasonable heuristics if we only want to check a small subset of stem bb combinations


## S2 model: putting it together

- increase seq len to `100`, but still fixed length (before we update training script for scoring network)

- increase dataset size to ???

- also generate evaluation dataset on chr1

### Generate dataset

- len=100, no threshold on mfe_freq

env `rna_ss_py3`

```
cd s2_data_gen
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py \
--len_min 100 --len_max 100 --num_seq 10000 --threshold_mfe_freq 0 \
--chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 \
--out ../data/human_transcriptome_segment_high_mfe_freq_training_len100_train_10000.pkl.gz
```


```
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py \
--len_min 100 --len_max 100 --num_seq 1000 --threshold_mfe_freq 0 \
--chromosomes chr1 \
--out ../data/human_transcriptome_segment_high_mfe_freq_training_len100_test_1000.pkl.gz
```

- run S1 inference

env `pytorch_plot_py3`

```
cd ../
taskset --cpu-list 21,22,23,24 python run_s1_inference_run32_ep79.py \
--data data/human_transcriptome_segment_high_mfe_freq_training_len100_train_10000.pkl.gz \
--threshold 0.5 --out data/data_len100_train_10000_s1_stem_bb.pkl.gz
```


```
taskset --cpu-list 21,22,23,24 python run_s1_inference_run32_ep79.py \
--data data/human_transcriptome_segment_high_mfe_freq_training_len100_test_1000.pkl.gz \
--threshold 0.5 --out data/data_len100_test_1000_s1_stem_bb.pkl.gz
```

### Generate all valid stem bb combinations

debug:

```
python run_s2_stem_bb_combo_tree_search.py --data data/data_len100_test_1000_s1_stem_bb.pkl.gz \
--out data/data_len100_test_1000_s1_stem_bb_combos.pkl.gz
```

Seems to be running forever with 92 bbs? O_O!



## S2 model: putting it together - second try

- increase seq len to `60`, but still fixed length (before we update training script for scoring network)

- also generate evaluation dataset on chr1

### Generate dataset

- len=60, no threshold on mfe_freq

env `rna_ss_py3`

```
cd s2_data_gen
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py \
--len_min 60 --len_max 60 --num_seq 10000 --threshold_mfe_freq 0 \
--chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 \
--out ../data/human_transcriptome_segment_high_mfe_freq_training_len60_train_10000.pkl.gz
```


```
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py \
--len_min 60 --len_max 60 --num_seq 1000 --threshold_mfe_freq 0 \
--chromosomes chr1 \
--out ../data/human_transcriptome_segment_high_mfe_freq_training_len60_test_1000.pkl.gz
```

- run S1 inference

env `pytorch_plot_py3`

```
cd ../
taskset --cpu-list 21,22,23,24 python run_s1_inference_run32_ep79.py \
--data data/human_transcriptome_segment_high_mfe_freq_training_len60_train_10000.pkl.gz \
--threshold 0.5 --out data/data_len60_train_10000_s1_stem_bb.pkl.gz
```


```
taskset --cpu-list 11,12,13,14 python run_s1_inference_run32_ep79.py \
--data data/human_transcriptome_segment_high_mfe_freq_training_len60_test_1000.pkl.gz \
--threshold 0.5 --out data/data_len60_test_1000_s1_stem_bb.pkl.gz
```

### Generate all valid stem bb combinations


```
taskset --cpu-list 11,12,13,14 python run_s2_stem_bb_combo_tree_search.py --data data/data_len60_test_1000_s1_stem_bb.pkl.gz \
--out data/data_len60_test_1000_s1_stem_bb_combos.pkl.gz
```

dataset statistics (len=60, num_bbs v.s. num_valid_bb_combos):

![plot/dataset_num_bb_combos_len60.png](plot/dataset_num_bb_combos_len60.png)

(plot produced by [check_bb_combo_statistics.ipynb](check_bb_combo_statistics.ipynb))

Observation: compare to last week (len=40), we now have significantly more valid combos,
even for the same number of bbs (x-axis), which is expected.





### Scoring network

- save scoring network model and parameters (not siamese network!)



## Var length



## MCTS

pre-trained score network

DP??

which node to add next should depend on:
(1) value of the next state,
(2) embedding from node constraint graph? and given current assignment? (similar to Khalil 2017)


## Read paper

2017



