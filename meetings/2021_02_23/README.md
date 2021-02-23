
## CVAE

### Derivation

Jamboard, page 5 for the correct version: https://jamboard.google.com/d/1EEPdX0WBT-af9FpjLmuF38E0D40D_85l_tsJMUHaLC8/viewer?f=4

Write p(y|x) interms of the q distribution.

Note:
original (wrong) version of graphical model: x -> z -> y,
should be x + z -> y

Gaussian closed form KLD: page 6 https://jamboard.google.com/d/1EEPdX0WBT-af9FpjLmuF38E0D40D_85l_tsJMUHaLC8/viewer?f=5


### Toy example


MNIST toy dataset

x -> 50% ground truth label, 50% ground truth label + 1 mod 10.
To construct the dataset, we make two copies of each example in minibatch,
one with original ground truth label, the other with ground truth label + 1 mod 10,
and pass in both in the same batch.

model architecture, training: page 7 of https://jamboard.google.com/d/1EEPdX0WBT-af9FpjLmuF38E0D40D_85l_tsJMUHaLC8/viewer?f=6



Result: https://docs.google.com/presentation/d/1eVUYtTeyt76zrLD3fQdQ_tQ3b0ExvXFrfh4fEo_ygVY/edit#slide=id.p

Observations:

- Model is learning to predict the correct stochatstic output, although far from being perfect.

- We rarely see 50/50 predicted distribution (which is how we've constructed the training).

- Most of the examples have predictions peaked at one of the two target labels, as can be seen
from the entropy histogram (for which a perfect predictor would peak at 0.69).


To reproduce the above, see [cvae_mnist.ipynb](cvae_mnist.ipynb)






## Read paper

### DeepSets

### DeepSetNet: Predicting Sets with Deep Neural Networks

### Joint Learning of Set Cardinality and State Distribution

### BRUNO: A Deep Recurrent Model for Exchangeable Data


### Deep Set Prediction Networks

## Datasets

(be careful: old dataset using top left corner format)

`6PvUty`: rnastralign?

`903rfx`: rfam151?

`a16nRG`: s_processed?

`xs5Soq`: synthetic?

`ZQi8RT`: synthetic? with prediction?

`DmNgdP`: bpRNA?

Sources:

- S1 training data, synthetic sequences: `ZQi8RT`


Intermediate:


## TODOs

- s2 inference: sampling mode, instead of taking argmax at each step (including the starting bb), sample w.r.t. the model output probability

- latent variable model

- when do we predict 'no structure'?

- try a few more params for S1 comparison plot: (1) t=0.02, k=1,c=0, (2) t=0.1,k=0,c=0.9, (3) t=0.1,k=0,c=0.5, ….etc.
generate another random test dataset (use new data format with top right corner)
try t=0.000001
try t=0.000001 and k=2


- s2 idea: stacked 2D map: seq + binary, one for each local structure (predicted by s1). self attn across 2d maps?

- s2 idea: GNN? 'meta' node connects local structure? predict on/off of meta node? still can't incoportate non-local structure

- dataset: '../2020_11_24/data/rfam151_s1_pruned.pkl.gz'  'data/synthetic_s1_pruned.pkl.gz'

- inference pipeline debug + improvement: n_proposal_norm > 1, implementation using queue, terminate condition

- s2 training: stems only? how to pass in background info like sequence? memory network? encoding?

- s2 training dataset, for those example where s1 bb sensitivity < 100%, add in the ground truth bbs for contructing dataset for s2.
How to set features like median_prob and n_proposal_norm? Average in the same example?

- rfam151 (and other dataset): evaluate base pair sensitivity and specificity (allow off by 1?)

- evaluate sensitivity if we allow +/-1 shift/expand of each bb

- if above works and we have a NN for stage 2, we can feed in this extended set of bb proposals!

- stage 1 prevent overfitting (note that theoretical upper bound is not 100% due to the way we constructed the predictive problem)

- investigate pseudoknot predictions, synthetic dataset (45886-32008)

- try running RNAfold and allow C-U and U-U (and other) base pairs, can we recover the lower FE structure that our model predicts?

- rfam151 dataset debug, is the ground truth bounding box correct? (make sure there’s no off-by-1 error)

- stage 1 model: iloop size = 0 on my side is bulge, make sure we have those cases!

- RNAfold performance on rfam151

- RNA-RNA interaction? Run stage 1 model three times, A-A, B-B & A-B, 2nd stage will have different constraints





old dataset in top left corner format, convert everything to top right?




