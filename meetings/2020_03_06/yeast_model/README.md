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

see https://github.com/PSI-Lab/alice-sandbox/blob/879b6d151ce0f949cb5386b4ef3eb3cdba1dc577/meetings/2020_03_06/yeast_model/investigate_data.ipynb

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

see https://github.com/PSI-Lab/alice-sandbox/blob/879b6d151ce0f949cb5386b4ef3eb3cdba1dc577/meetings/2020_03_06/yeast_model/investigate_data.ipynb

## Check correlation between fitness and interaction in raw data

NxN:
```
df['interaction'].corr(df['fd'])
0.2743129604320869
```

see https://github.com/PSI-Lab/alice-sandbox/blob/879b6d151ce0f949cb5386b4ef3eb3cdba1dc577/meetings/2020_03_06/yeast_model/investigate_data.ipynb

## Train NN to predict all 3 fitness scores

This doesn't make sense.
If we provide double and single fitness score as training targets,
then the training dataset will contain all single fitness scores,
then the NN can learn to memorize those,
which makes everything trivial, and is kinda of cheating.

## Can we prediction single KO fitness?

Does learning fitness on gene A, B, C tell us anything about gene D, E, F?

Ill-posed problem? Since D, E, F will be constant input in training set,
NN will ignore them.

Seems to be an impossible problem without extra information.

Is it possible to learn a structure so that it generalizes?



Hmm, if NN jointly predicts fitness and interaction,
does it mean it somewhat has access to all 3 fitness scores?
Is it possible it can memorize all single fitness?
If that's the case then the problem might be trivial?
- maybe not

## Multi-task learning without providing single fitness

Train model to predict too targets,
double fitness and interaction score.
Internally the NN has 3 parallel modules, with shared weights,
takes input double KO, single KO a, single KO b, and
predicting double fitness, single fitness a, single fitness b, respectively.
One loss function connects double fitness to the fitness target.
On the other hand, one extra layer with hard-coded logic and no parameter
compute gi = fd - fa * fb, on top of which we compute the loss between predicted and target gi.
Such setup does not provide single ko fitness (no cheating),
but still encourage to learn something in common.

Does this work?


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


TODO is 0/1 encoding the best? try 1/-1??


