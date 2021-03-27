

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



dataset


```
python s2_training/make_dataset_sequence_encoding.py --in_file ../2021_03_23/data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned.pkl.gz --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_encode_sequence.npz
```

TODO no need to remove non-100% sensitivity examples.





## TODOs

- S1 inference: (1) mask lower triangular for all softmax! (2) get rid of invalid stem (there could be many if we don’t threshold on probability)

- check training set metric on ep_50 (overfit) to see if it’s better?

- S2 dataset: running S1 inference got rid of more than 50% examples? compare with thresholding on on_off probability


