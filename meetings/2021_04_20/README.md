
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


## S1 prediction


TODO re-run S1 pred with both methods combined?

see [tmp_s1_pred_stem_processing.ipynb](tmp_s1_pred_stem_processing.ipynb)

Converted to script:

```
python s1_pred_stem_processing.py --data ../2021_03_23/data/debug_training_len20_200_100.pkl.gz \
--threshold_p 0.1 --threshold_n 0.5 --model ../2021_03_23/s1_training/result/run_7/model_ckpt_ep_17.pth \
--out_file data/debug.pkl.gz
```

Large dataset:

```
python s1_pred_stem_processing.py --data ../2021_03_23/data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000.pkl.gz \
--threshold_p 0.1 --threshold_n 0.5 --model ../2021_03_23/s1_training/result/run_7/model_ckpt_ep_17.pth \
--out_file data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz
```




## S1 non-conv model

TODO


## Read papers on GNN edge labelling/pruning

### Edge-Labeling Graph Neural Network for Few-shot Learning

###﻿EdgeNets:Edge Varying Graph Neural Networks

### Neural Relational Inference for Interacting Systems


## S2 GNN idea

From last week:

![plot/s2_gnn_idea.png](plot/s2_gnn_idea.png)


Other advantages:

- S2 model will be able to 'fine-tune' some bbs, e.g. make it 1bp smaller (won't be able to extend it)

debug:


```
python s2_train_gnn_1.py --input_data data/debug_training_len20_200_100_s1_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.01 --epochs 1 --batch_size 10 --hid 10 10 --log tmp.log
```


real dataset:


```
python s2_train_gnn_1.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.01 --epochs 1 --hid 10 10 --log tmp.log
```

```
python s2_train_gnn_1.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.001 --epochs 100 --hid 20 20 20 20 20 20 20 20 20 20 --log result/s2_gnn_run_1.log
```


```
python s2_train_gnn_1.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.0005 --epochs 100 --hid 50 50 50 50 50 50 50 50 50 50 --log result/s2_gnn_run_2.log
```

```
python s2_train_gnn_1.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.0005 --epochs 100 --hid 200 200 200 200 --batch_size 20 --log result/s2_gnn_run_3.log
```


### k-mer embedding

```
python s2_train_gnn_2.py --input_data data/debug_training_len20_200_100_s1_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.01 --epochs 1 --batch_size 10 --hid 10 10 --log tmp.log --kmer 3 --embed_dim 20
```


```
python s2_train_gnn_2.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.001 --epochs 100 --batch_size 10 --hid 50 50 50 50 50 \
 --log result/s2_gnn_run_4.log --kmer 5 --embed_dim 100
```

overfit?

```
2021-04-18 01:39:45,737 [MainThread  ] [INFO ]  Epoch 99, training, mean loss 0.017095582736049882, mean AUC 0.7990723658650051
2021-04-18 01:39:48,782 [MainThread  ] [INFO ]  Epoch 99, testing, mean loss 0.6578992579819104, mean AUC 0.5951351727922218
```

#### try smaller NN

```
python s2_train_gnn_2.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps.pkl.gz \
--training_proportion 0.95 --learning_rate 0.001 --epochs 100 --batch_size 10 --hid 20 20 20 20 20 \
 --log result/s2_gnn_run_5.log --kmer 3 --embed_dim 50
```

better?

```
2021-04-18 09:59:59,187 [MainThread  ] [INFO ]  Epoch 99, training, mean loss 0.02624728402795171, mean AUC 0.7104701549247721
2021-04-18 10:00:00,974 [MainThread  ] [INFO ]  Epoch 99, testing, mean loss 0.27273789339558374, mean AUC 0.6843603798848465
```


### TODOs


not working?

- make it easier problem: (1) shorter seq, (2) fewer proposals?

- read paper

- k-mer embedding

- edge feature 0, 1 -> [0, 1] & [1, 0]

- per-node softmax (on the matrix, using mask?)

- debug loss accumulation

- GPU

- (done) sub-class graph conv layer, add in edge features, added GATEConv

- (done, seems to help with performance but not sure if it's overfitting) concat GCN features across multiple layers

- (done, checked that above is overfitting on small data) add validation set

- simple inner product

- toy task 2: predict base pair binary label

- more layers (more hops - we should cover the size of a typical 'loop')

- other toy tasks?

- convert edge to node? like a factor graph?

- real dataset (more examples)


- node level prediction: whether each base is paired


### Gradient check for loss accumulation

See [gradient_check.ipynb](gradient_check.ipynb)


## S2 GNN - toy task

To test the implementation, we put together a toy dataset as follows:

Graph is being constructed in the same way:

- nodes are the bases

- two types of edges:

    - backbone connection: undirected edges connects i and i+1, for 0 <= i < seq_len

    - putative hydrogen bonds from S1 stem bb proposal: undirected edges connecting all base pairs in predicted stems

Edge targets are constructed to be trivial:
An edge is labelled as 1 if the base pair is G-C (or C-G), and 0 otherwise.
Any reasonable model should be able to achieve close to 100% AUC on this dataset.

Model architecture:

- 1 layer of graph convolution (graph attention convolution)

- outer concatenate node features after graph conv

- 1x1 convolution (equivalent to FC for each node pair)

- sigmoid output with BCE loss

Result:

```
Epoch 0, mean loss 0.6725149569766862, mean AUC 0.6756173759394002
Epoch 1, mean loss 0.6480377220681736, mean AUC 0.9750365592911423
Epoch 2, mean loss 0.5782683294798646, mean AUC 0.9929219380802792
Epoch 3, mean loss 0.428862168852772, mean AUC 0.9985667497092067
Epoch 4, mean loss 0.24169315450957843, mean AUC 0.9997053254384118
Epoch 5, mean loss 0.12639604535486018, mean AUC 0.9999354404240066
Epoch 6, mean loss 0.07895537938124367, mean AUC 0.9999469424885264
Epoch 7, mean loss 0.058380261595760076, mean AUC 0.9999701805995238
Epoch 8, mean loss 0.047000310400367847, mean AUC 0.9999729680559698
Epoch 9, mean loss 0.040260948573372195, mean AUC 0.9999848190217681
```

Generated by: [tmp_gnn_toy_task.ipynb](tmp_gnn_toy_task.ipynb)


## S2 GNN implementation

### Dataset




To generate training dataset for S2, we run S1 model (combined inferences?),
and keep those example whose stem bb sensitivity is 100%.

The following annotations are generated for each example:

- sequence

- length

- base pair indices for all stem bbs

- target base pair indices


TODO:

- sensitivity for identical bb

- sensitivity for: predicted bb >= target bb

- add in 'missing' bb for s2


### Feature encoding


Node feature

Edge feature


### Training

target

link?






## TODOs

use only stem bb proposals, initialize GNN connection,
(one base could be connected to multiple ones), predict final edge weights (each node can have 0 or 1 edge),
(but we lose bb?)


PSI meeting TODOs:
LSTM debug: gradient? is it learning anything? check the weight connecting the LSTM outputs
S2 baseline: decrease NN capacity
S2 baseline: try different subsets of input features (bb type, location, size)
S2 baseline naive guess? metric? (but it’s actually pretty clear from the shuffled-label experiment that it’s not predicting all 0's or the naive guess)
S2 other ideas involving ‘background sequence’?

randomize features?

backpropr to s1?

memory network

other ways to summerize s1 pred?

s2: BERT like? predict masked bbs?

s2: sampling -> combination -> rank using a score function (this should be trivial)

s1+2:  global attn -> predict bb?  auxilary target and mask in lower layers (start with cnn?),
last layer has to predict the non-masked bbs. combine all bb and average pixels?

read https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html#ssd-single-shot-multibox-detector

read https://openreview.net/forum?id=snOgiCYZgJ7

s1: mimic FC (instead of CNN), use only one layer of 2D conv with large receptive field (50x50?),
all subsequent layers are 1x1 2D conv (FC)

multi-resolution?


learn how to combine pixel's predictions



s1 inference: combine both? sensitivity? n proposal?

s1 predict: on/off distribution (per example), percentile?

-S 1 inference: check likelihood of ground truth bb


- S1 inference: (1) mask lower triangular for all softmax! (2) get rid of invalid stem (there could be many if we don’t threshold on probability)

- check training set metric on ep_50 (overfit) to see if it’s better?

- S2 dataset: running S1 inference got rid of more than 50% examples? compare with thresholding on on_off probability


