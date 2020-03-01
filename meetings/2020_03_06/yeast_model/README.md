Recap from last week:

After chatting with Andrew, we realized that the 0.5 correlation reported in the paper
was from predicting interaction, not fitness.
Our performance dropped significantly after switching to training on gene interaction as target.
Our test performance is around 0.4 right now (haven't done extensive hyperparameter tuning).
So it's kind of in line with the paper.

According to Albi from DG, people typically calculate interaction score
as: fitness_ab - fitness_a * fitness_b (need to read original paper to confirm),
so it's very different than predicting just a double KO fitness.
We need to at least be able to also predict single KO fitness.
Albi also mentioned that, a better CV split should be by gene pairs, e,g, A & B in training, C & D in validation
(how can the model generalize in this case? needs further discussion).


## Investigate relationship between single KO fitness and double KO fitness

Verified that interaction score was calculated as:

```
f1, f2, fd: fd - f1 * f2
```

TODO how did they compute the std?

## Check D-cell performance on interaction

For the downloaded file 'costonzo_2009_pvalue005_prediction_growth',
although the name says 'growth', its correlation with double KO fitness is low.
Also checked against interaction score, low as well.

For the downloaded file 'costonzo_2009_pvalue005_prediction',
the numbers are more reasonable:

ExN:
```
df_dcell_pred['pred'].corr(df_dcell_pred['interaction'])
0.31460551573959655
df_dcell_pred['pred'].corr(df_dcell_pred['fd'])
0.22879967021468772
```


NxN:
```
df_dcell_pred['pred'].corr(df_dcell_pred['interaction'])
0.4892228685092307
df_dcell_pred['pred'].corr(df_dcell_pred['fd'])
0.21512434519724072
```


## Check correlation between fitness and interaction in raw data

NxN:
```
df['interaction'].corr(df['fd'])
0.2743129604320869
```

## Can we prediction single KO fitness?

Does learning fitness on gene A, B, C tell us anything about gene D, E, F?



## Train NN to predict all 3 fitness scores


TODO how can one gene generalize to another?

TODO how can gene pair A, B generalize to gene pair C, D?

- understand why gene interaction is more difficult (& important) than fitness

- hyperparam tuning to match their performance

- multitask: predict fitness and interaction?

- incorporate other prior information to improve performance


- read cell map paper, make sure we understand how gene interaction were calculated,
and why it is more important than fitness


- dataset target value distribution, correlation between reps

- try prediction fitness of single gene KO

- KO v.s. KD, different alleles



