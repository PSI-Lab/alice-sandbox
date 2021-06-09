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

- len=100, no threshold on mfe_freq:

env `rna_ss_py3`

```
cd s2_data_gen
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py \
--len_min 100 --len_max 100 --num_seq 10000 --threshold_mfe_freq 0 \
--chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 \
--out ../data/human_transcriptome_segment_high_mfe_freq_training_len100_train_10000.pkl.gz
```


```
cd s2_data_gen
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py \
--len_min 100 --len_max 100 --num_seq 1000 --threshold_mfe_freq 0 \
--chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 \
--out ../data/human_transcriptome_segment_high_mfe_freq_training_len100_test_1000.pkl.gz
```


## Var length



## MCTS

pre-trained score network

DP??




## Read paper

2017



