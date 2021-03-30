

## S1 prediction pruning

### Debug

out-of-bound bb filtering not working? -> checked code, tested on toy example, seems to be ok, re-running

```
python model_utils/prune_stage_1.py --in_file ../2021_03_23/data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000.pkl.gz --discard_ns_stem --min_hloop_size 3 --min_pixel_pred 0 --min_prob 0 --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned.pkl.gz
```

Above data works. Confirmed no bugs in code.


## S2 LSTM + self-attention

### Encode sequence feature

- model ideas: S2 with LSTM as sequence encoder. Jamboard: https://jamboard.google.com/d/1exf2uP4ZERQDsTx2A2zEugMcZ8F-vGjMue2KsnuCyC8/viewer?f=0

- Note that in the above encoding,
there's actually intrisic symmetry (e.g. flip for stem),
which could potentially be modeled in a systematic fashion in NN.

- Data processing for training


debug

```
python s2_training/make_dataset_sequence_encoding.py --in_file ../2021_03_23/data/debug_s1_pred_len20_200_100_pruned.pkl.gz --out_file data/debug_s1_pred_len20_200_100_pruned_filtered_encode_sequence.npz
```



dataset 5000


```
python s2_training/make_dataset_sequence_encoding.py --in_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned.pkl.gz --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_encode_sequence.npz
```




TODO no need to remove non-100% sensitivity examples.


### training

- add bb-specific LSTM (using pytorch to pack and unpack sequence)

debug

```
cd s2_training/
python train_lstm_self_attention.py --in_file ../data/debug_s1_pred_len20_200_100_pruned_filtered_encode_sequence.npz --config debug_config.yml --out_dir result/debug/
```


dataset 5000


```
cd s2_training/
CUDA_VISIBLE_DEVICES=1 python train_lstm_self_attention.py --in_file ../data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_encode_sequence.npz --config config_lstm_attn.yml --out_dir result/run_1/
```

### Baseline feature

Only bounding box location and size. This is to investigate the performance gap
between the model using S1 features and the model using LSTM sequence encoder.
We expect model using LSTM sequence encoder to be at least better than this baseline.

debug

```
python s2_training/make_dataset_baseline.py --in_file ../2021_03_23/data/debug_s1_pred_len20_200_100_pruned.pkl.gz --out_file data/debug_s1_pred_len20_200_100_pruned_filtered_baseline.npz
```

dataset 5000

```
python s2_training/make_dataset_baseline.py --in_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned.pkl.gz --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_baseline.npz
```


### training


debug

```
cd s2_training/
python train_s2.py --in_file ../data/debug_s1_pred_len20_200_100_pruned_filtered_baseline.npz --config config_baseline.yml --out_dir result/debug/
```

dataset 5000

```
cd s2_training/
CUDA_VISIBLE_DEVICES=2 python train_s2.py --in_file ../data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_baseline.npz --config config_baseline.yml --out_dir result/run_baseline_1/
```

### Debug - baseline features with shuffled labels

target binary label shuffled. For debug.

debug

```
python s2_training/make_dataset_debug_shuffle_label.py --in_file ../2021_03_23/data/debug_s1_pred_len20_200_100_pruned.pkl.gz --out_file data/debug_s1_pred_len20_200_100_pruned_filtered_debug_shuffle_label.npz
```

dataset 5000

```
python s2_training/make_dataset_debug_shuffle_label.py --in_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned.pkl.gz --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_debug_shuffle_label.npz
```


### Training


debug

```
cd s2_training/
python train_s2.py --in_file ../data/debug_s1_pred_len20_200_100_pruned_filtered_debug_shuffle_label.npz --config config_baseline.yml --out_dir result/debug/
```

dataset 5000

```
cd s2_training/
CUDA_VISIBLE_DEVICES=2 python train_s2.py --in_file ../data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_debug_shuffle_label.npz --config config_baseline.yml --out_dir result/run_baseline_debug_shuffle_label_1/
```


### Result

- bb location, size & S1 feature (n_proposal_norm, median probability): training au-ROC 0.98 validation 0.96


- bb location, size & LSTM sequence encoding joint training


    - LSTM hid=20, layer=4, 50 epochs: training au-ROC 0.878 validation 0.87

    - LSTM hid=50, layer=4, 50 epochs: training au-ROC 0.887 validation 0.88

    - LSTM hid=100, layer=10, 50 epochs: training au-ROC 0.889 validation 0.886

    - LSTM hid=100, layer=10, 100 epochs: training au-ROC 0.9 validation 0.886

- bb location, size (baseline): training au-ROC 0.93 validation 0.87

Why is baseline performance so high?
To make sure there's no bug in the code, we generated a debug
dataset with shuffled labels:

- bb location, size with shuffled label (debug, we expect random performance):
training au-ROC 0.5 validation 0.5  (re-assuring)



## TODOs

- S1 inference: (1) mask lower triangular for all softmax! (2) get rid of invalid stem (there could be many if we don’t threshold on probability)

- check training set metric on ep_50 (overfit) to see if it’s better?

- S2 dataset: running S1 inference got rid of more than 50% examples? compare with thresholding on on_off probability


