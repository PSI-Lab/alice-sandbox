Last week:

- after fixing the batchnorm bug, we tried a few modelling ideas,
mainly focused on making training pairs less trivial to improve generalization performance

- some performance gain was achieved by carefully constructing the distribution
of 'negatives' in the siamese pair, see table from last week

- we've also checked whether training on a contrastive learning objective (siamese network,
1 v.s. other, which is actually a quite difficult (in terms of generalization) form of constrastive learning objective)
enables the scoring network to learn the actual ordering.
We've plotted the predicted score v.s. RNAfold FE on selected examples, see last week's plot.

- Last week we were evaluating on a subset of test set (where s1 sensitivity is 100%),
we'll switch to the full set this week


## Evaluating on full test set

- use full test set with 1000 sequences

- report the following:

    - subset to examples where the target structure is within top k=100, ranked by n_bps

    - subset to examples where target structure has FE <= -25, use scoring network to predict top k=100 (ranked by n_bps),
    report f1 score of the best out of best-10 predicted

    - repeat above with parameter combinations: FE: {-25, -15, inf} x best-n: {10, 5, 3, 1}

| f1                                  | run_11   | run_12   | run_13   | run_14   | run_15   |
|-------------------------------------|----------|----------|----------|----------|----------|
| target in topK, best-1 predicted    |          |          |          |          |          |
| mean                                | 0.680618 | 0.73928  | 0.738815 | 0.722802 | 0.630028 |
| median                              | 0.881944 | 0.947368 | 0.9375   | 0.947368 | 0.843206 |
| target FE <= -25, best-1 predicted  |          |          |          |          |          |
| mean                                | 0.533702 | 0.572872 | 0.585713 | 0.546648 | 0.527177 |
| median                              | 0.480769 | 0.511905 | 0.676984 | 0.511905 | 0.511905 |
| target FE <= -25, best-3 predicted  |          |          |          |          |          |
| mean                                | 0.630908 | 0.652533 | 0.660674 | 0.643172 | 0.628257 |
| median                              | 0.650407 | 0.676984 | 0.746429 | 0.697516 | 0.676984 |
| target FE <= -25, best-5 predicted  |          |          |          |          |          |
| mean                                | 0.663246 | 0.701061 | 0.694731 | 0.646363 | 0.695936 |
| median                              | 0.704762 | 0.781685 | 0.786063 | 0.697516 | 0.795671 |
| target FE <= -25, best-10 predicted |          |          |          |          |          |
| mean                                | 0.717604 | 0.734488 | 0.737752 | 0.749256 | 0.750541 |
| median                              | 0.824891 | 0.824891 | 0.838877 | 0.868687 | 0.838877 |
| target FE <= -15, best-1 predicted  |          |          |          |          |          |
| mean                                | 0.490091 | 0.628985 | 0.490824 | 0.512334 | 0.494961 |
| median                              | 0.5      | 0.628985 | 0.519969 | 0.518315 | 0.525063 |
| target FE <= -15, best-3 predicted  |          |          |          |          |          |
| mean                                | 0.591948 | 0.628985 | 0.585754 | 0.613344 | 0.591867 |
| median                              | 0.647059 | 0.628985 | 0.637073 | 0.666667 | 0.647059 |
| target FE <= -15, best-5 predicted  |          |          |          |          |          |
| mean                                | 0.639195 | 0.628985 | 0.635757 | 0.644539 | 0.62488  |
| median                              | 0.704293 | 0.694208 | 0.70778  | 0.702703 | 0.701351 |
| target FE <= -15, best-10 predicted |          |          |          |          |          |
| mean                                | 0.699421 | 0.691921 | 0.69604  | 0.69506  | 0.686605 |
| median                              | 0.777778 | 0.764706 | 0.781746 | 0.769231 | 0.766968 |
| all, best-1 predicted               |          |          |          |          |          |
| mean                                | 0.423986 | 0.435495 | 0.424741 | 0.435468 | 0.412325 |
| median                              | 0.421053 | 0.434783 | 0.422648 | 0.4375   | 0.4      |
| all, best-3 predicted               |          |          |          |          |          |
| mean                                | 0.534541 | 0.530035 | 0.527244 | 0.545004 | 0.519846 |
| median                              | 0.583333 | 0.578947 | 0.567766 | 0.588235 | 0.567766 |
| all, best-5 predicted               |          |          |          |          |          |
| mean                                | 0.582198 | 0.581566 | 0.573895 | 0.590041 | 0.572703 |
| median                              | 0.631579 | 0.636364 | 0.625    | 0.64611  | 0.628571 |
| all, best-10 predicted              |          |          |          |          |          |
| mean                                | 0.646293 | 0.641669 | 0.632936 | 0.64821  | 0.637174 |
| median                              | 0.7      | 0.695652 | 0.6875   | 0.704293 | 0.697826 |


- we also checked FE v.s. predicted score (from top k=100),
note that we need to remove those outlier data points where RNAfold predicts a super high FE (pseudoknot)

| corr     | run_11   | run_12   | run_13   | run_14   | run_15   |
|----------|----------|----------|----------|----------|----------|
| pearson  |          |          |          |          |          |
| mean     | 0.417005 | 0.404382 | 0.462412 | 0.462149 | 0.408052 |
| median   | 0.416107 | 0.434081 | 0.442211 | 0.479674 | 0.455443 |
| spearman |          |          |          |          |          |
| mean     | 0.359852 | 0.354028 | 0.410240 | 0.394711 | 0.352610 |
| median   | 0.352739 | 0.383931 | 0.418311 | 0.412303 | 0.375027 |

Above produced by:

```
mkdir -p result/eval_s2_log
python eval_s2_full_test_set.py --run_id run_11 > result/eval_s2_log/run_11.txt; \
python eval_s2_full_test_set.py --run_id run_12 > result/eval_s2_log/run_12.txt; \
python eval_s2_full_test_set.py --run_id run_13 > result/eval_s2_log/run_13.txt; \
python eval_s2_full_test_set.py --run_id run_14 > result/eval_s2_log/run_14.txt; \
python eval_s2_full_test_set.py --run_id run_15 > result/eval_s2_log/run_15.txt
```


visualize selected examples?

TODO plot




todo: score network is unnormalized, raw score can all be very high/low

todo: as a proof-of-concept, train score network on FE directly?

theoretical upper bound given S2 model (use RNAeval, also use topk? but with a big k)


Q learning
