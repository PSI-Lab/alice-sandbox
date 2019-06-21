## Introduction

## Trained Models

Trained models have been uploaded to DataCorral:

- fold 0: `olGFJp`

- fold 1: `DmsqY9`

- fold 2: `G4BMyS`

- fold 3: `ZvyR2t`

- fold 4: `bGnMPk`


## Performance Report

### Cross validation performance on all transcripts

Here we compute Pearson and Spearman correlation on the entire training dataset, using cross validation models (each data point is being predicted using the model that wasn't trained on it):

This plot shows a distribution of correlation values across different training/validation examples.
Each example is a 5000bp sequence-value pair, on which we compute the correlation between target value and prediction.
Missing target values are being ignored.

[plot/cross_validation_performance.html](plot/cross_validation_performance.html)

Summary:

```
           pearson     spearman
count  3147.000000  3147.000000
mean      0.467310     0.482307
std       0.189137     0.194578
min      -1.000000    -1.000000
25%       0.354125     0.366015
50%       0.473358     0.497206
75%       0.596210     0.618222
max       0.999861     1.000000
```

### Comparison to RNAplfold on a subset of transcripts

Transcripts were selected by Omar, see https://github.com/deepgenomics/omar-sandbox/tree/master/rnaplfold_vs_probing

Each plot shows a distribution of correlation values across different exons in the selected transcripts, for a particular flanking context length.
To predict on exons in a particular transcript, we use the model that wasn't trained on that transcript.
Flanking context length is being used to feed the model, as well as being passed to RNAplfold for computing binding probability.
Correlation is computed only on bases within the exon. Missing target values are being ignored.

- Flank=50

[plot/compare_rnafold_flank_50.html](plot/compare_rnafold_flank_50.html)

Summary:

```
       pearson_corr_model  pearson_corr_rnafoldr  spearman_corr_model  spearman_corr_rnafoldr
count          144.000000             144.000000           144.000000              144.000000
mean             0.850485               0.394820             0.861628                0.410172
std              0.049123               0.159697             0.047838                0.156227
min              0.630301              -0.046478             0.668074               -0.030199
25%              0.818994               0.296204             0.836182                0.329875
50%              0.863315               0.401978             0.871584                0.416588
75%              0.882479               0.503662             0.896020                0.517376
max              0.947462               0.758301             0.944314                0.808930
```

- Flank=100

[plot/compare_rnafold_flank_100.html](plot/compare_rnafold_flank_100.html)

Summary:

```
       pearson_corr_model  pearson_corr_rnafoldr  spearman_corr_model  spearman_corr_rnafoldr
count          142.000000             142.000000           142.000000              142.000000
mean             0.853867               0.405132             0.865054                0.415437
std              0.045190               0.146610             0.044249                0.146330
min              0.711402              -0.135376             0.704844               -0.177987
25%              0.821643               0.320862             0.839319                0.332192
50%              0.861718               0.404053             0.872388                0.423581
75%              0.883304               0.509033             0.896491                0.512532
max              0.946470               0.741699             0.945529                0.800446
```

- Flank=150

[plot/compare_rnafold_flank_150.html](plot/compare_rnafold_flank_150.html)

Summary:

```
       pearson_corr_model  pearson_corr_rnafoldr  spearman_corr_model  spearman_corr_rnafoldr
count          139.000000             139.000000           139.000000              139.000000
mean             0.853533               0.402768             0.865417                0.409189
std              0.045549               0.144315             0.044284                0.142757
min              0.711558              -0.223389             0.704844               -0.163276
25%              0.820982               0.328369             0.839699                0.320437
50%              0.861491               0.414062             0.871579                0.427567
75%              0.883343               0.498291             0.896760                0.506163
max              0.946424               0.761230             0.945529                0.800446
```

- Flank=200

[plot/compare_rnafold_flank_200.html](plot/compare_rnafold_flank_200.html)

Summary:

```
       pearson_corr_model  pearson_corr_rnafoldr  spearman_corr_model  spearman_corr_rnafoldr
count          138.000000             138.000000           138.000000              138.000000
mean             0.854562               0.401551             0.866296                0.412697
std              0.044064               0.146213             0.043211                0.140997
min              0.728586               0.016525             0.704844                0.022655
25%              0.821539               0.308289             0.840276                0.316303
50%              0.861731               0.414673             0.872388                0.428394
75%              0.883374               0.495300             0.896841                0.503001
max              0.946424               0.772949             0.945529                0.768871
```

- Flank=250

[plot/compare_rnafold_flank_250.html](plot/compare_rnafold_flank_250.html)

Summary:

```
       pearson_corr_model  pearson_corr_rnafoldr  spearman_corr_model  spearman_corr_rnafoldr
count          135.000000             135.000000           135.000000              135.000000
mean             0.854131               0.402980             0.865914                0.413017
std              0.044224               0.144429             0.043552                0.142156
min              0.728586              -0.009094             0.704844               -0.011929
25%              0.820982               0.316284             0.839699                0.327221
50%              0.861491               0.414062             0.871579                0.417296
75%              0.882998               0.492310             0.896760                0.512036
max              0.946424               0.733398             0.945529                0.804276
```

## Reproduce the models

To generate data and train model:

```bash
mkdir -p data
mkdir -p model
python make_data.py
CUDA_VISIBLE_DEVICES={gpu_id} python train.py 0
CUDA_VISIBLE_DEVICES={gpu_id} python train.py 1
CUDA_VISIBLE_DEVICES={gpu_id} python train.py 2
CUDA_VISIBLE_DEVICES={gpu_id} python train.py 3
CUDA_VISIBLE_DEVICES={gpu_id} python train.py 4
mkdir -p plot
CUDA_VISIBLE_DEVICES={gpu_id} python evaluate.py
```

Most parameters are defined in [config.py](config.py).

Note that in the current version of config we're only trianing/validating on chr1-22, so the processed data (gtrack) don't have any data points for transcripts on other chromosomes.
As a result, one transcript `NM_000291` is dropped in the evaluation. This can be updated in future rounds if we add in chrX/Y.

You only need to rerun `python make_data.py` if you've changed the way the gtracks are being generated.


## Future Work

- (shreshth) reduce LR should have lower patience than ES. See train.py line #116. Also set a min_lr at 1e-6

- (shreshth) Some people have reported gated linear units working better than gated tanh units on some tasks. I don't have any intuition or preference, but may be worth trying.
Ref. https://arxiv.org/abs/1612.08083

- re-weight training examples by transcript coverage

- train on other dataset (cap reactivity at 1)

- multi-task on all dataset (need to reprocess rep1/2 to combine into one value)

- read other papers in Omar's list, check if there's any low throughput data we can use for validation

- sync up with Mark in 1-2 weeks regarding processed data from Matthew's lab

- clean up and check in re-implementation of DMfold (also try train/validation split by RNA type,
to avoid 'cheating' since there are highly similar sequences within the same RNA type)

- (Amit) try different ways of N encoding [0,0,0,0] v.s. [0.25,0.25,0.25,0.25]

- predictor sequence padding, make it optional, (implement interval interface in modelzoo (real seq / N's / real + N's when real req is not sufficient for context))

- re-process raw data so it's fully GK compatible
