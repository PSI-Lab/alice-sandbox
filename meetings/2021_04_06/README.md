
## Investigate S2 baseline model

Following last week's strange result: https://github.com/PSI-Lab/alice-sandbox/tree/8ba16db7867e16e11831af62b447cf5b69ad84bc/meetings/2021_03_30#result,
we'd like to run a few more experiments to understand:
(1) whether there's any bug that's causing the high performance of baseline model,
and if not, (2) how can baseline model achieve such high performance?

Baseline model: 7 input features from 3 categories:
1-hot encoded bb type (3) + bb location (2)
+ bb size (2).

### Without bb type

Dataset:

debug

```
python s2_training/make_dataset_baseline_no_bb_type.py --in_file ../2021_03_23/data/debug_s1_pred_len20_200_100_pruned.pkl.gz --out_file data/debug_s1_pred_len20_200_100_pruned_filtered_baseline_no_bb_type.npz
```

dataset 5000


```
python s2_training/make_dataset_baseline_no_bb_type.py --in_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned.pkl.gz --out_file data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_baseline_no_bb_type.npz
```


Training:

debug

```
cd s2_training/
python train_s2.py --in_file ../data/debug_s1_pred_len20_200_100_pruned_filtered_baseline_no_bb_type.npz --config config_baseline_no_bb_type.yml --out_dir result/debug/
```

dataset 5000

```
cd s2_training/
CUDA_VISIBLE_DEVICES=2 python train_s2.py --in_file ../data/human_transcriptome_segment_high_mfe_freq_s1_pred_len20_200_5000_pruned_baseline_no_bb_type.npz --config config_baseline_no_bb_type.yml --out_dir result/run_baseline_np_bb_type_1/
```


## Investigate S1 LSTM+sttn model

 TODO gradient, weights?



## TODOs

PSI meeting TODOs:
LSTM debug: gradient? is it learning anything? check the weight connecting the LSTM outputs
S2 baseline: decrease NN capacity
S2 baseline: try different subsets of input features (bb type, location, size)
S2 baseline naive guess? metric? (but it’s actually pretty clear from the shuffled-label experiment that it’s not predicting all 0's or the naive guess)
S2 other ideas involving ‘background sequence’?


-S 1 inference: check likelihood of ground truth bb


- S1 inference: (1) mask lower triangular for all softmax! (2) get rid of invalid stem (there could be many if we don’t threshold on probability)

- check training set metric on ep_50 (overfit) to see if it’s better?

- S2 dataset: running S1 inference got rid of more than 50% examples? compare with thresholding on on_off probability



