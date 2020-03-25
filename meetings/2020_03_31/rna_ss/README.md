
Implement res block:

debug run:
```
python train_nn.py --data /Users/alicegao/work/psi-lab-sandbox/rna_ss/data_processing/rnafold_mini_data/data/rand_seqs_var_len_5_20_10.pkl.gz --result result/debug --num_filters 16 32 64 --num_stacks 2 2 2 --epoch 10 --batch_size 500 --cpu 8
```

rnafold data:
```
CUDA_VISIBLE_DEVICES=1 python train_nn.py --data /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/rnafold_mini_data/make_data_var_len/data/rand_seqs_var_len_sample_mfe_10_200_1000000.pkl.gz --result result/debug --num_filters 16 32 64 --num_stacks 2 2 2 --epoch 10 --batch_size 500 --cpu 8
```


chain datasets (different length grouping)

leaky relu

dilation

res blocks

bottleneck layers

other tricks?



SPOT-RNA model

max length 500nt,



block-wise prediction -> models co-transcriptional folding?


train on non-simulated data


