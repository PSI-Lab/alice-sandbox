
Paper: "Mapping the genetic landscape of human cells"
Data downloaded from https://ars.els-cdn.com/content/image/1-s2.0-S0092867418307359-mmc5.xlsx


using 2nd tab "gene GI scores and correlations", 
hand-selected columns "K562 Replicate Average GI score" and "Jurkat Replicate Average GI score".


Input: 2-hot encoding of gene KO, output: GI scores for K562 & Jurkat

Number of training examples: 57304, test: 14327

Number of genes: 378

LR:
```bash
python train_nn.py --data gene_gi_scores.csv.gz --result result/run_lr --epoch 100 --hid_sizes
```
Train:
```
2020-03-04 10:08:20,009 [MainThread  ] [INFO ]  correlation k562
2020-03-04 10:08:20,011 [MainThread  ] [INFO ]  (0.13213352889614388, 1.727386187929456e-221)
2020-03-04 10:08:20,012 [MainThread  ] [INFO ]  correlation jurkat
2020-03-04 10:08:20,013 [MainThread  ] [INFO ]  (0.10326285219124257, 1.2990708891036103e-135)
```
Test:
```
2020-03-04 10:08:20,325 [MainThread  ] [INFO ]  correlation k562
2020-03-04 10:08:20,326 [MainThread  ] [INFO ]  (0.1202070887210314, 2.92507908343209e-47)
2020-03-04 10:08:20,326 [MainThread  ] [INFO ]  correlation jurkat
2020-03-04 10:08:20,326 [MainThread  ] [INFO ]  (0.08914620975162568, 1.1245307933826678e-26)
```

NN:
```bash
CUDA_VISIBLE_DEVICES=2 python train_nn.py --data gene_gi_scores.csv.gz --result result/run_1 --epoch 100 --hid_sizes 50 20 10
```
Train:
```
2020-03-03 08:48:51,128 [MainThread  ] [INFO ]  correlation k562
2020-03-03 08:48:51,138 [MainThread  ] [INFO ]  (0.576343182791189, 0.0)
2020-03-03 08:48:51,138 [MainThread  ] [INFO ]  correlation jurkat
2020-03-03 08:48:51,139 [MainThread  ] [INFO ]  (0.41553700187891635, 0.0)
```
Test:
```
2020-03-03 08:48:51,490 [MainThread  ] [INFO ]  correlation k562
2020-03-03 08:48:51,491 [MainThread  ] [INFO ]  (0.46848966263356473, 0.0)
2020-03-03 08:48:51,491 [MainThread  ] [INFO ]  correlation jurkat
2020-03-03 08:48:51,492 [MainThread  ] [INFO ]  (0.29029019909554477, 3.353243299003869e-276)
```


TODOs:








