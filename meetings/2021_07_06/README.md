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




todo: score network is unnormalized, raw score can all be very high/low

todo: as a proof-of-concept, train score network on FE directly?


