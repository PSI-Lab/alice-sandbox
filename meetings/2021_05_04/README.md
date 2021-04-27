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
