## Debug - dataset

From last week:

Some dataset entries have almost identical sequences?

```
df.iloc[1].seq
'CCAGCAACTGCTGGCCTGTGCCAGGGTGCAAGCTGAGCACTGGAGTGGAGTTTTCCTGTGGAGA'
df.iloc[2].seq
'CAGCAACTGCTGGCCTGTGCCAGGGTGCAAGCTGAGCACTGGAGTGGAGTTTTCCTGTGGAGAG'
```


Some consecutive pairs have low edit distance:

```
for i in range(len(df)):
    for j in range(i+1, len(df)):
        distance = levenshtein(df.iloc[i].seq, df.iloc[j].seq)
        if distance <= 10:
            print(i, j, distance)

1 2 2
27 28 2
29 30 6
54 55 6
93 94 8
```


To be further investigated.




## S1 non-conv model

TODO


## Read papers on GNN edge labelling/pruning

### Edge-Labeling Graph Neural Network for Few-shot Learning

###﻿EdgeNets:Edge Varying Graph Neural Networks

### Neural Relational Inference for Interacting Systems



## S2 GNN Per-node softmax


From last week:

- instead of predicting the binary label for each edge,
predict the picked edge for each node

- output mask should be multiplied before taking softmax

- implemented on top of previous step (i.e. with kmer embedding)

- Caveat: we're not taking care of the case where a node has no connection
(target will be `[0, 0, ..., 0]`, will the gradient work in this case?).
Proper solution would be to predict another per-node binary label,
and mask out softmax gradient for those nodes where ground truth is no connection.


debug:

```
python s2_train_gnn_3.py --input_data ../2021_04_20/data/debug_training_len20_200_100_s1_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.01 --epochs 1 --batch_size 10 --hid 10 10 --log tmp.log --kmer 3 --embed_dim 20
```


real data:

```
python s2_train_gnn_3.py --input_data ../2021_04_20/data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.001 --epochs 100 --batch_size 10 --hid 20 20 20 20 20 \
 --log result/s2_gnn_run_3_1.log --kmer 3 --embed_dim 50
```

Reporting accuracy:


```
2021-04-21 18:56:32,590 [MainThread  ] [INFO ]  Epoch 99, training, mean loss 0.4069409890287727, mean accuracy 0.3617328405380249
2021-04-21 18:56:34,138 [MainThread  ] [INFO ]  Epoch 99, testing, mean loss 4.129653316636325, mean accuracy 0.36053499579429626
```

increase capacity (also fixed loss scale):

```
python s2_train_gnn_3.py --input_data ../2021_04_20/data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.001 --epochs 100 --batch_size 10 --hid 50 50 50 50 50 \
 --log result/s2_gnn_run_3_2.log --kmer 3 --embed_dim 50
```

### TODOs


not working?

- fix 10x loss

- modified GATEconv to make it simple (just node + edge feature)

- make it easier problem: (1) shorter seq, (2) fewer proposals?

- read paper

- jamboard

- re-try softmax: update eval metric to accuracy (not global auc)

- update GATEconv -> simpler, just use edge feature

- super node?

- backbone edge directed?

- try using S1 pred. see if it improves performance? (although we wouldn't want to use it in actual training)

- debug dataset, inspect a few examples, make sure it make sense

- plot, distribution of: num_edges per node, v,s, len?

- incoportate s1 prediction as edge features

- GPU

- modify GATEconv?

- simple inner product

- toy task 2: predict base pair binary label

- more layers (more hops - we should cover the size of a typical 'loop')

- other toy tasks?

- convert edge to node? like a factor graph?

- real dataset (more examples)


- node level prediction: whether each base is paired



