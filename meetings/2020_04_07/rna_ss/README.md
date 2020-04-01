

debug: try e2efold dataset, set max length (for now) to avoid GPU OOM

```
CUDA_VISIBLE_DEVICES=0 python train_nn.py --data /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/e2efold/data/rnastralign.pkl.gz --result result/run_1 --num_filters 16 32 32 64 --num_stacks 4 4 4 4 --epoch 10 --batch_size 20 --max_length 200 --cpu 8
```


train:

remove redundant sequences: how?

use same train/validation/test fold

report same metric: precision, recall, F1
(did they pick a threshold then computing those metric?)

read E2Efold paper

how was their performance without the second network?

how did they manage momery for long sequence (up to a few thousand bases)
