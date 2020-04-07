

debug: try e2efold dataset, set max length (for now) to avoid GPU OOM

```
CUDA_VISIBLE_DEVICES=0 python train_nn.py --data /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/e2efold/data/rnastralign.pkl.gz --result result/run_1 --num_filters 16 32 32 64 --num_stacks 4 4 4 4 --epoch 10 --batch_size 20 --max_length 200 --cpu 8
```



E2Efold training dataset (only using sequences <= 500bp):

```
2020-04-01 16:52:42,202 [MainThread  ] [DEBUG]  Cmd: Namespace(batch_size=2, cpu=8, data=['/home/alice/work/psi-lab-sandbox/rna_ss/data_processing/e2efold/data/rnastralign.pkl.gz'], epoch=10, max_length=500, num_filters=[16, 32, 32, 64], num_stacks=[4, 4, 4, 4], result='result/run_2')
2020-04-01 16:52:42,211 [MainThread  ] [DEBUG]  Current dir: /home/alice/work/psi-lab-sandbox/meetings/2020_04_07/rna_ss, git hash: b'149a296aab4256c87083de480265d14e80640818'
2020-04-01 16:52:42,211 [MainThread  ] [INFO ]  Loading dataset: ['/home/alice/work/psi-lab-sandbox/rna_ss/data_processing/e2efold/data/rnastralign.pkl.gz']
2020-04-01 16:52:43,212 [MainThread  ] [INFO ]  Subsetting to max length, n_rows before: 37138
2020-04-01 16:52:43,329 [MainThread  ] [INFO ]  After: 25309
2020-04-01 16:52:44,765 [MainThread  ] [INFO ]  Using 24043 data for training and 1266 for validation
2020-04-01 16:53:17,581 [MainThread  ] [INFO ]  Naive guess: 0.003552073612809181
2020-04-01 16:53:17,581 [MainThread  ] [INFO ]  Naive guess performance
2020-04-01 16:55:16,660 [MainThread  ] [INFO ]  Training: loss 282.77260564214833 au-ROC 0.5 au-PRC 0.0052068848438962655
2020-04-01 16:55:23,119 [MainThread  ] [INFO ]  Validation: loss 276.6368779186954 au-ROC 0.5 au-PRC 0.005243201182008695

2020-04-01 17:26:59,847 [MainThread  ] [INFO ]  Epoch 0/10, training loss (running) 248.17947429139926, au-ROC 0.8567060957232291, au-PRC 0.4153584170082467
2020-04-01 17:27:29,800 [MainThread  ] [INFO ]  Epoch 0/10, validation loss 248.17947429139926, au-ROC 0.9944128459038376, au-PRC 0.870200835242462
2020-04-01 17:48:21,379 [MainThread  ] [INFO ]  Epoch 1/10, training loss (running) 58.34958481463889, au-ROC 0.995238934018685, au-PRC 0.8963399403645365
2020-04-01 17:48:51,168 [MainThread  ] [INFO ]  Epoch 1/10, validation loss 58.34958481463889, au-ROC 0.9978768479652509, au-PRC 0.9444577674908223
2020-04-01 18:09:52,787 [MainThread  ] [INFO ]  Epoch 2/10, training loss (running) 33.12656814523687, au-ROC 0.9983264730838959, au-PRC 0.9523307390088513
2020-04-01 18:10:22,515 [MainThread  ] [INFO ]  Epoch 2/10, validation loss 33.12656814523687, au-ROC 0.9987776995698857, au-PRC 0.960123459140409
2020-04-01 18:45:06,192 [MainThread  ] [INFO ]  Epoch 3/10, training loss (running) 26.030567330879197, au-ROC 0.9988769210819045, au-PRC 0.9648309144724957
2020-04-01 18:46:02,510 [MainThread  ] [INFO ]  Epoch 3/10, validation loss 26.030567330879197, au-ROC 0.9990366039700443, au-PRC 0.9661389078030683
2020-04-01 19:26:12,220 [MainThread  ] [INFO ]  Epoch 4/10, training loss (running) 22.138253928977587, au-ROC 0.9991587680885341, au-PRC 0.9712408607415305
2020-04-01 19:27:09,764 [MainThread  ] [INFO ]  Epoch 4/10, validation loss 22.138253928977587, au-ROC 0.999238202696707, au-PRC 0.973661540475827
2020-04-01 20:08:07,274 [MainThread  ] [INFO ]  Epoch 5/10, training loss (running) 19.57962425424874, au-ROC 0.9993185871780057, au-PRC 0.9754736919534104
2020-04-01 20:09:04,942 [MainThread  ] [INFO ]  Epoch 5/10, validation loss 19.57962425424874, au-ROC 0.999283302456531, au-PRC 0.9739151255405293
2020-04-01 20:50:00,247 [MainThread  ] [INFO ]  Epoch 6/10, training loss (running) 17.71732763997743, au-ROC 0.9994420549721759, au-PRC 0.9784852071635047
2020-04-01 20:50:57,467 [MainThread  ] [INFO ]  Epoch 6/10, validation loss 17.71732763997743, au-ROC 0.999271676415518, au-PRC 0.9761044338193131
```




train:

remove redundant sequences: how?

use same train/validation/test fold

report same metric: precision, recall, F1
(did they pick a threshold then computing those metric?)
(use sklearn https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

read E2Efold paper

how was their performance without the second network?

how did they manage momery for long sequence (up to a few thousand bases)
