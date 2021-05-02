## Summary

### Finished + WIP


### Ideas to try


## Re-generated dataset S1 pred

- continued from last week


## S1 pred prob_on only with S1 features

- continued from last week


- update s1 inference and s2 data gen to include all examples,
regardless of whether it's 100% sensitivity

### Data generation with all examples

debug:

```
python s1_pred_stem_processing.py --data ../2021_03_23/data/debug_training_len20_200_100.pkl.gz \
--threshold_p 0.1 --model ../2021_03_23/s1_training/result/run_7/model_ckpt_ep_17.pth \
--out_file data/debug_only_prob_on.pkl.gz --features --include_all
```

real data:

```
taskset --cpu-list 21,22,23,24 python s1_pred_stem_processing.py --data ../2021_03_23/data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000.pkl.gz \
--threshold_p 0.1 --model ../2021_03_23/s1_training/result/run_7/model_ckpt_ep_17.pth \
--out_file data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps_only_prob_on_all.pkl.gz \
--features  --include_all
```

data generated at

```
sandbox/meetings/2021_05_04/data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps_only_prob_on_all.pkl.gz
```

### Training

- same hyperparameter as last week (2nd largest model), with new dataset:

- added model export

debug:

```
python s2_train_gnn_10.py --input_data data/debug_only_prob_on.pkl.gz \
--training_proportion 0.8 --learning_rate 0.01 --epochs 2 --batch_size 10 --hid 20 20 \
 --log result/debug.log --kmer 3 --embed_dim 10
```


real data:

```
taskset --cpu-list 1,2,3,4 python s2_train_gnn_10.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps_only_prob_on_all.pkl.gz \
--training_proportion 0.95 --learning_rate 0.001 --epochs 200 --batch_size 10 --hid 20 20 20 20 20 50 50 50 50 50 100 100 \
 --log result/s2_gnn_run_10_6.log --kmer 3 --embed_dim 50
```

Validation au-ROC around 0.71 now:


```
2021-04-28 12:52:05,853 [MainThread  ] [INFO ]  Epoch 199, training, mean loss 0.408006290952533, mean AUC 0.7671182695147436
2021-04-28 12:52:05,866 [MainThread  ] [INFO ]  Model checkpoint saved at: result/s2_gnn_run_10_6.model_ckpt_ep_199.pth
2021-04-28 12:52:18,937 [MainThread  ] [INFO ]  Epoch 199, testing, mean loss 0.6002969351126326, mean AUC 0.7133638415491533
```




- added prediction export for debugging


debug:

```
python s2_train_gnn_10.py --input_data data/debug_only_prob_on.pkl.gz \
--training_proportion 0.8 --learning_rate 0.01 --epochs 2 --batch_size 10 --hid 20 20 \
 --log result/debug.log --kmer 3 --embed_dim 10 --debug
```

experiment with larger LR, more hid, less epochs:


real data:

```
taskset --cpu-list 5,6,7,8,9,10 python s2_train_gnn_10.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps_only_prob_on_all.pkl.gz \
--training_proportion 0.95 --learning_rate 0.01 --epochs 100 --batch_size 10 --hid 20 20 40 40 40 50 50 50 50 100 100 100 \
 --log result/s2_gnn_run_10_7.log --kmer 3 --embed_dim 50 --debug
```


- also add export x (will need to construct one-off obj (data.x, data.edge_index, data.edge_attr) for debugging afterwards)

debug:

```
python s2_train_gnn_10.py --input_data data/debug_only_prob_on.pkl.gz \
--training_proportion 0.8 --learning_rate 0.01 --epochs 2 --batch_size 10 --hid 20 20 \
 --log result/debug.log --kmer 3 --embed_dim 10 --debug
```

real data:

```
taskset --cpu-list 5,6,7,8,9,10 python s2_train_gnn_10.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps_only_prob_on_all.pkl.gz \
--training_proportion 0.95 --learning_rate 0.01 --epochs 100 --batch_size 10 --hid 20 20 40 40 40 50 50 50 50 100 100 100 \
 --log result/s2_gnn_run_10_8.log --kmer 3 --embed_dim 50 --debug
```

Above exported data to be debugged.


Increase capacity, lower LR a bit:

```
taskset --cpu-list 1,2,3,4 python s2_train_gnn_10.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps_only_prob_on_all.pkl.gz \
--training_proportion 0.95 --learning_rate 0.005 --epochs 100 --batch_size 10 --hid 20 20 40 40 40 40 50 50 50 50 50 100 100 100 \
 --log result/s2_gnn_run_10_9.log --kmer 3 --embed_dim 50
```

Better!

```
2021-04-29 17:29:11,456 [MainThread  ] [INFO ]  Epoch 99, training, mean loss 0.40581630037750227, mean AUC 0.7929436711922307
2021-04-29 17:29:29,108 [MainThread  ] [INFO ]  Epoch 99, testing, mean loss 0.43014416960348567, mean AUC 0.7762389614882788
2021-04-29 17:29:29,163 [MainThread  ] [INFO ]  Model checkpoint saved at: result/s2_gnn_run_10_9.model_ckpt_ep_99.pth
```

Further increase capacity, lower LR a bit:

```
taskset --cpu-list 1,2,3,4 python s2_train_gnn_10.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps_only_prob_on_all.pkl.gz \
--training_proportion 0.95 --learning_rate 0.002 --epochs 100 --batch_size 10 --hid 20 20 40 40 40 40 50 50 50 50 50 50 100 100 100 100 \
 --log result/s2_gnn_run_10_11.log --kmer 3 --embed_dim 50
```


a bit overfitting?

```
2021-04-30 19:14:09,323 [MainThread  ] [INFO ]  Epoch 99, training, mean loss 0.35730640315529655, mean AUC 0.8230427627751322
2021-04-30 19:14:28,643 [MainThread  ] [INFO ]  Epoch 99, testing, mean loss 0.5197086364711564, mean AUC 0.7722066496235012
2021-04-30 19:14:28,710 [MainThread  ] [INFO ]  Model checkpoint saved at: result/s2_gnn_run_10_11.model_ckpt_ep_99.pth
```



### Debug

Model prediction on validation set saved during training is different from
prediction made using loaded model? O_O!!

Doesn't look like all the parameters are being saved?

```
model_params.keys()

odict_keys(['node_embedding.weight', 'node_pair_conv1.weight', 'node_pair_conv1.bias', 'node_pair_conv2.weight', 'node_pair_conv2.bias'])
```

Need to use `nn.ModuleList` instead of Python list! (This is the 2nd time I'm making this mistake!)

Fixed! (TODO fix other scripts, update shared repo)

Re-run after fixing:

```
taskset --cpu-list 5,6,7,8,9,10 python s2_train_gnn_10.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_pred_stem_bps_only_prob_on_all.pkl.gz \
--training_proportion 0.95 --learning_rate 0.001 --epochs 100 --batch_size 10 --hid 20 20 40 40 40 50 50 50 50 100 100 100 \
 --log result/s2_gnn_run_10_10.log --kmer 3 --embed_dim 50 --debug
```

done:

```
2021-04-30 12:52:16,838 [MainThread  ] [INFO ]  Epoch 99, training, mean loss 0.32630523789863464, mean AUC 0.8375920197869435
2021-04-30 12:52:28,172 [MainThread  ] [INFO ]  Epoch 99, testing, mean loss 0.8460604499134174, mean AUC 0.7748649759824477
2021-04-30 12:52:28,221 [MainThread  ] [INFO ]  Model checkpoint saved at: result/s2_gnn_run_10_10.model_ckpt_ep_99.pth
2021-04-30 12:52:31,369 [MainThread  ] [INFO ]  Prediction saved at: result/s2_gnn_run_10_10.pred_va_ep_99.pkl
```


#### Verify the fix

- From run `s2_gnn_run_10_10`, we've saved both the model parameters, as well as input and prediction on validation set

- We initialize the model, load the parameters, and use the model to compute prediction on one randomly picked example

- Compare the above with the predictions saved during training

- We've verified the two arrays are almost equal:

```
np.testing.assert_almost_equal(tmp_pred.detach().numpy(), data['yp'], decimal=5)
```

- Visualizing a small slice of the array, we can see that they're almost equal:

```
tmp_pred[:2, :2].detach().numpy()
array([[1.1315030e-22, 4.6394876e-21],
       [2.4324624e-16, 9.9738324e-15]], dtype=float32)

data['yp'][:2, :2]
array([[1.1314901e-22, 4.6394165e-21],
       [2.4324716e-16, 9.9737562e-15]], dtype=float32)
```

- Now we've verified that the model parameters have been fully saved and restored

See [tmp_debug.ipynb](tmp_debug.ipynb)

### Eval

performance mismatch? during training v.s. loaded model.. debugging

## Re-generate dataset

- sciprts from `2021_03_23`

- updated to avoid generating overlapping sequences

- note this needs GK


Dataset 5000 (after fix overlapping):

```
cd s2_data_gen
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py --len_min 20 --len_max 200 --num_seq 5000 --threshold_mfe_freq 0.02 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_2.pkl.gz
```

Run inference:


```
taskset --cpu-list 21,22,23,24 python s1_pred_stem_processing.py --data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_2.pkl.gz \
--threshold_p 0.1 --model ../2021_03_23/s1_training/result/run_7/model_ckpt_ep_17.pth \
--out_file data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_2_pred_stem_bps_only_prob_on_all.pkl.gz \
--features  --include_all
```

done



### Re-training

```
taskset --cpu-list 1,2,3,4 python s2_train_gnn_10.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_5000_2_pred_stem_bps_only_prob_on_all.pkl.gz \
--training_proportion 0.95 --learning_rate 0.002 --epochs 100 --batch_size 10 --hid 20 20 40 40 40 40 50 50 50 50 50 50 100 100 100 100 \
 --log result/s2_gnn_run_10_12.log --kmer 3 --embed_dim 50
```



## Generate a larger dataset for training



debug

```
cd s2_data_gen
python generate_human_transcriptome_segment_high_mfe_freq_var_len.py --len_min 20 --len_max 200 --num_seq 10 --threshold_mfe_freq 0.02 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/debug_training_len20_200_10.pkl.gz
```




Dataset 20000:

```
cd s2_data_gen
taskset --cpu-list 21,22,23,24 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py --len_min 20 --len_max 200 --num_seq 20000 --threshold_mfe_freq 0.02 --chromosomes chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 --out ../data/human_transcriptome_segment_high_mfe_freq_training_len20_200_20000.pkl.gz
```

Run inference:


```
taskset --cpu-list 21,22,23,24 python s1_pred_stem_processing.py --data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_20000.pkl.gz \
--threshold_p 0.1 --model ../2021_03_23/s1_training/result/run_7/model_ckpt_ep_17.pth \
--out_file data/human_transcriptome_segment_high_mfe_freq_training_len20_200_20000_pred_stem_bps_only_prob_on_all.pkl.gz \
--features  --include_all
```

done.

### Training

```
taskset --cpu-list 21,22,23,24 python s2_train_gnn_10.py --input_data data/human_transcriptome_segment_high_mfe_freq_training_len20_200_20000_pred_stem_bps_only_prob_on_all.pkl.gz \
--training_proportion 0.95 --learning_rate 0.002 --epochs 100 --batch_size 10 --hid 20 20 40 40 40 40 50 50 50 50 50 50 100 100 100 100 \
 --log result/s2_gnn_run_10_13.log --kmer 3 --embed_dim 50
```

running

## TODOs

- TODO copy from last week

- TODO debug data gen? similar sequences?

- bpRNA S2

- predict binary label for each node

- add bb location as feature? how? or positional encoding of nodes?

- residual connection?

- double check S1 doesn't have the same parameter saving problem

- larger dataset?

- check video

- bpRNA: GNN predict distance? process raw data?

