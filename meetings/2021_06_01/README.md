Last week:

- tried training one model on new S2 dataset (with updated S1 model), performance is similar
as before, ~0.45 overall F1 score, which is not very good (given dataset was generated in a
deterministic way)

- run S1 inference with different thresholds,
observed tradeoff between bb sensitivity and connection density


## S2 model re-training

- from last week: running S1 inference with threshold = 0.5,
dataset generated at: `/home/alice/work/psi-lab-sandbox/meetings/2021_05_25/s2_train_len20_200_20000_pred_stem_0p5.pkl.gz`

- from last week: this dataset has mean density = `0.043`
and S1 sensitivity:

```
   threshold  bb_identical_mean  bb_identical_std  bb_overlap_mean  bb_overlap_std   bp_mean    bp_std
0        0.5           0.815131          0.189959         0.907873          0.1353  0.919245  0.110961
```

- update training script to report more metrics

- model training (fewer epoch):

```
taskset --cpu-list 1,2,3,4 python s2_train_gnn_10.py --input_data ../2021_05_25/s2_train_len20_200_20000_pred_stem_0p5.pkl.gz \
--training_proportion 0.95 --learning_rate 0.002 --epochs 20 --batch_size 10 --hid 20 20 40 40 40 40 40 50 50 50 50 50 50 50 100 100 100 100 100 \
 --log result/s2_gnn_run_30.log --kmer 3 --embed_dim 50
```



TODOs

- thoughts: GNN architecture

- predict p(x), basepair probability,
local bb informative for this target? need to run RNAfold and generate this data.

- base S2 model, no GNN update,
just node 1 feature + node 2 feature -> target,
can be done by outer product -> 1x1 2D conv, mask gradient

-double check GATEConv

- sigmoid + softmax

- add position as feature

- discrete, constraint enforcement

- algorithmic NN? find paper







