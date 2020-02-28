

## Debug

Shuffle target value (for training not for testing), and inspect the performance:

https://github.com/PSI-Lab/alice-sandbox/tree/9ed0d6968cbef4d647489341604d5fd71cb741c5/meetings/2020_02_28/yeast_model_training

```
python train_base_line_nn.py --data ~/work/psi-lab-sandbox/meetings/2020_02_11/d_cell/data/training/raw_pair_wise/SGA_ExE.txt --result result/exe_shuffle_target --epoch 20 --shuffle_label
```

run log: https://github.com/PSI-Lab/alice-sandbox/blob/5243edffa09d8daf85a39bb57b597b5be2014859/meetings/2020_02_28/yeast_model_training/result/exe_shuffle_target/run.log


Without shuffling (we already ran it last time, but re-running it here anyways):

```
python train_base_line_nn.py --data ~/work/psi-lab-sandbox/meetings/2020_02_11/d_cell/data/training/raw_pair_wise/SGA_ExE.txt --result result/exe_no_shuffle --epoch 20
```

run log: https://github.com/PSI-Lab/alice-sandbox/blob/5243edffa09d8daf85a39bb57b597b5be2014859/meetings/2020_02_28/yeast_model_training/result/exe_no_shuffle/run.log

Comparing the two run logs, and notice the following differences:

- After shuffling target values for the training set,
performance drop to that similar of a naive solution
(which makes sense, since the only gradient direction that's consistent across all minibatches,
 should be pointing towards the naive solution),
and it's consistent between training and testing

## New findings

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



## TODOs

- understand why gene interaction is more difficult (& important) than fitness

- hyperparam tuning to match their performance

- multitask: predict fitness and interaction?

- incorporate other prior information to improve performance

- check correlation between gene interaction and prediction:
http://chianti.ucsd.edu/~kono/ci/data/deep-cell/predictions/prediction.tar.gz

- read cell map paper, make sure we understand how gene interaction were calculated,
and why it is more important than fitness


- dataset target value distribution, correlation between reps

- try prediction fitness of single gene KO

- KO v.s. KD, different alleles



