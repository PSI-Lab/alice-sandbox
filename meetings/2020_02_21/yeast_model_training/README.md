# Reproduce D-cell paper

Previous work: https://github.com/PSI-Lab/alice-sandbox/tree/3dd7dac1f1fdcc146600e8465a69c298f119da23/meetings/2020_02_11

To reproduce the result, first download the raw data from http://thecellmap.org/costanzo2016/data_files/Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip,
unzip and put all files in a `data/` folder.

- Essential x Essential genes

LR performance:

https://github.com/PSI-Lab/alice-sandbox/tree/34476c7663564a49358648781a6d766d4be0c306/meetings/2020_02_21/yeast_model_training

```
python train_base_line_nn.py --data ~/work/psi-lab-sandbox/meetings/2020_02_11/d_cell/data/training/raw_pair_wise/SGA_ExE.txt --result result/exe_lr --epoch 100
```

See https://github.com/PSI-Lab/alice-sandbox/blob/e40ee0512fb523ec5b791cb42f6188ada92743ef/meetings/2020_02_21/yeast_model_training/result/exe_lr/run.log

NN performance:



```

```

- NE x E


- NE x NE


- All genes



