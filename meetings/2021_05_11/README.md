
## Fix target label during training

Last week: when target bb is not identical to but within a predicted bb,
we exclude that target bb when generating target, where we should have included
a subset of the base pairs that are covered by predicted bbs.


debug:

```
python s2_train_gnn_10.py --input_data ../2021_05_04/data/debug_only_prob_on.pkl.gz \
--training_proportion 0.8 --learning_rate 0.01 --epochs 2 --batch_size 10 --hid 20 20 \
 --log result/debug.log --kmer 3 --embed_dim 10
```



TODO missing link prediction (only if we do not depend on edge features)


