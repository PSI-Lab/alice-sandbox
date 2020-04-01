


```
CUDA_VISIBLE_DEVICES=0 python train_nn.py --data /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/e2efold/data/rnastralign.pkl.gz --result result/run_3 --num_filters 16 32 32 64 --num_stacks 4 4 4 4 --epoch 10 --batch_size 20 --max_length 200 --cpu 8
```
