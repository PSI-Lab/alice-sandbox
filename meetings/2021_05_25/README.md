
Last week:

- updated s1 model to only predict stem bbs

- updated S1 dataset to have multiple structures per sequence,
to better capture all 'plausible' stem bbs

- trained a couple of S1 models with different hyperparameters

- best model so far: `../2021_05_18/s1_training/result/run_32/model_ckpt_ep_79.pth`,
picked

- s2 model: generated dataset 5k & 20k, run s1 inference, started S2 model training


## S2 model

- from last week (ran from 2021_05_18):

```
taskset --cpu-list 21,22,23,24 python s2_train_gnn_10.py --input_data ../2021_05_18/data/s2_train_len20_200_5000_pred_stem.pkl.gz \
--training_proportion 0.95 --learning_rate 0.002 --epochs 100 --batch_size 10 --hid 20 20 40 40 40 40 40 50 50 50 50 50 50 50 100 100 100 100 100 \
 --log result/s2_gnn_run_20.log --kmer 3 --embed_dim 50
```


running


- test set 1000

```
cd s2_data_gen
taskset --cpu-list 11,12,13,14 python generate_human_transcriptome_segment_high_mfe_freq_var_len.py \
--len_min 20 --len_max 200 --num_seq 1000 --threshold_mfe_freq 0.02 \
--chromosomes chr1 \
--out ../data/human_transcriptome_segment_high_mfe_freq_test_len20_200_1000.pkl.gz
```

```
taskset --cpu-list 11,12,13,14 python s1_pred_stem_processing.py --data ../2021_05_18/data/human_transcriptome_segment_high_mfe_freq_training_len20_200_20000.pkl.gz \
--threshold_p 0.1 --model ../2021_05_18/s1_training/result/run_32/model_ckpt_ep_79.pth \
--out_file data/s2_train_len20_200_20000_pred_stem.pkl.gz \
--features  --include_all
```



TODO generate test set on chr1, run s1 inference

TODO evaluation

TODO sigmoid + softmax



## Writing


## TODOs

- end-to-end hyperparameter tuning?
S1 inference threshold

- baseline comparison