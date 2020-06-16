

## SCFG

### ﻿Evaluation of several lightweight stochastic context-free grammars for RNA secondary structure prediction

﻿Robin D Dowell 2004

* Existing SCFG based approaches:

    - ﻿Knudsen and Hein:
    "﻿RNA Secondary Structure Prediction Using Stochastic Context-Free Grammars and Evolutionary History."
    & "﻿Pfold: RNA Secondary Structure Prediction Using Stochastic Context-Free Grammars."

    - ﻿Rivas E, Eddy SR: "Secondary Structure Alone is Generally Not Statistically Significant for the Detection of Noncoding RNAs."

    - ﻿Holmes I, Rubin GM: "Pairwise RNA Structure Comparison with
Stochastic Context-Free Grammars. "

    - ﻿Rivas E, Eddy SR: "Noncoding RNA Gene Detection Using Com- parative Sequence Analysis."

* This paper investigates 9 lightweight SCFGs

* A simple DP works out the structure scores for all sub sequences in an outward direction (increasing length),
where at each step we consider the cases for extension: base pair, unpair (either side)
and bifurcation (need to consider all intermediate points for bifurcation).

* A complex DP is fundamentally the same as above, with more energy parameters.

* Probabilistic approaches are essentially the same as well.
﻿SCFGs captures the long range, nested, pairwise correlations,
such as those induced by base pairing in non-pseudoknotted RNA secondary structures.
﻿The ability to generate or score two or more correlated symbols in a single step is what gives CFGs the power to deal with base pairing.

* ﻿An SCFG describes a joint probability distribution P(x, π|G, Θ) over all RNA sequences x and all possible parse trees π.
Given a parameterized SCFG (G, Θ) and a sequence x, the
Cocke-Younger-Kasami (CYK) dynamic programming algorithm finds an optimal (maximum probability) parse tree for a sequence x.


* CYK algorithm is nearly identical to the classic simple DP for this application.
Difference is that the scoring system is probabilistic,
based on factoring the score for a structure down into a sum of log probability terms,
rather than factoring the structure into a sum of energy terms or arbitrary base-pair scores.


* In SCFG, we use the Inside algorithm to find probability of the sequence given grammar (and parameters),
by summing over all possible parse trees.
This is analogous to ﻿the McCaskill algorithm for calculating the partition function in thermodynamic models.

* ﻿Any SCFG that will be useful for RNA secondary structure prediction must be ambiguous in the strict sense.
﻿The optimal parse tree gives us the optimal structure if
and only if there is a one to one correspondence between parse trees and secondary structures.
However, a given secondary structure does not necessarily have a unique parse tree.

* The paper investigated whether certain grammars are structurally ambiguous in an empirical way.

* SCFG, first order dependencies such as base stacking can be captured by ﻿lexicalization.

* Training data: ﻿European Ribosomal Database, ﻿278 sequences, 586,293 nucleotides, and 146,759 base pairs.

* Test dataset 1: ﻿Ribonuclease P database [49], the Signal Recognition Particle database [50] and the tmRNA database [51].
﻿403 total sequences, consisting of 225 RNase P's, 81 SRP's, and 97 tmRNA's.


* Test datset 2: ﻿Rfam v5.0 [52]. ﻿2455 sequences from 174 different RNA families.

* Compared with ﻿mfold v3.1.2, Pfold (Oct 2003), PKNOTS v 1.01, RNAstructure v4.0, and the Vienna RNA package vl.4

* They did a "reordering experiment":
For each sequence, find the optimal parse tree using CYK, also sample suboptimal parse trees from the posterior.
Within these suboptimal trees, find the structure corresponding to each tree,
then for each structure, calculate the probability by summing over all parse trees for that structure
via conditional inside algorithm.
They checked how much difference exist between the rank order of the trees v.s. the structures,
and whether the max prob structure is consistent with the max prob tree.
They show that the difference is significant for ambiguous grammars, so one has to use unambiguous grammar.

* They showed that some simple grammar performs surprisingly well,
e.g. G6 with only 21 free parameters. See Table 3.


* Making the grammar more complex by including stacking correlation parameters helped in some cases, but not all.

* Authors also mentioned that ﻿the ideal training set would be a large number of evolutionarily unrelated RNA secondary structures.



* Code & data (broken link): ﻿http://www.genetics.wustl.edu/eddy/publications/#DowellEddy04.


### ﻿RNA secondary structure prediction using stochastic context-free grammars and evolutionary history

﻿B. Knudsen and J. Hein, 1999

* Input: alignment of RNA sequences, output: one common structure

* Alignment represented by matrix, where rows = number of sequences, cols = number of alignment positions.
The probability of the alignment, given the phylogenetic tree T and model M, can be computed
by summing over all structures.
Probability of alignment given the structure, T and M, is simply the product of column probabilities.

* Grammar can be extended to include production of columns, then the above probability can
be computed using Inside-Outside algorithm.





### ﻿Pfold: RNA secondary structure prediction using stochastic context-free grammars

﻿Bjarne Knudsen and Jotun Hein, 2003


* improvedment over the above one (faster, more sequences in alignment)

### ﻿Knudsen and Hein

## RNA structure representation

### Adventures with RNA Graphs






## 3D structure

### ﻿An RNA Scoring Function for Tertiary Structure Prediction Based on Multi-Layer Neural Networks

﻿Y. Z. Wanga, 2
, J. Lia, 2
, S. Zhanga , B. Huanga, G. Yaoa , and J. Zhanga, *, 2019

* Input: RNA 3D structure, output: score.
Trained 2 models: coarse & atom-level

* ML part is a bit sketchy

* hand-crafted feature extraction on 3D structure,
mostly summary statistics about interacting bases/atoms.
Coarse model has 291 features and atom-level model has 5524 features.

* Coarse model: 291-30-1, atom-level model: 5524-10-1.
Tanh activation. Loss is weighted MSE (more weight on structures with lower energies)

* Dataset: real sequences and structures from PDB (paper says NDM but I think it's a typo), 462 RNAs.
Decoys generated by molecular dynamics simulation, 300 per each positive.



## Structured output

### ﻿Learning Structured Output Representation using Deep Conditional Generative Models

﻿Kihyuk Sohn∗† Xinchen Yan† Honglak Lee

* conditional VAE, ﻿models the distribution of high-dimensional output space
as a generative model conditioned on the input observation.
Not only to model the complex output representation but also to make a discriminative prediction.

* Conditional generative process:
﻿for given observation x, z is drawn from the prior distribution p(z|x),
and the output y is generated from the distribution p(y|x, z).

* ﻿trained to maximize the conditional log-likelihood.
Variational lower bound resembles that of the classic VAE, with added conditioning on x. See Eq (4) and (5).

* In practise, the architecture has built-in recurrent connect using the baseline CNN output y as input, see Fig 1d.

* To evaluate the model on single output y, we can perform deterministic inference by
either taking the expected value of z, or marginalizing over z by sampling.
For y, we can either take the argmax, or compute the log-likelihood.

* In practise, reconstruction of y (training task) is easier than predicting y (testing task).
Author mentioned that weighing the KL divergence term more did not help.
Instead, they proposed to set the ﻿recognition network the same as the prior network,
i.e., qφ(z|x, y) = pθ(z|x), which they termed ﻿Gaussian stochastic neural network (GSNN).
This can also be combined with CVAE objective for an hybrid approach.

* The effectiveness of using conditional stochastic latent variable was demonstrated by a couple of experiments.
See Fig 3 & 4. For Fig 4, they showed that with block-occluded input,
their model is able to sample realistic & sharp segmentation, as opposed to the blurred CNN baseline.


### ﻿Fully Convolutional Networks for Semantic Segmentation

﻿Evan Shelhamer∗, Jonathan Long∗, and Trevor Darrell, 2016

not relevant, does not seem to address output structure directly


### Link Prediction with Cardinality Constraint

Jiawei Zhang?
, Jianhui Chen†
, Junxing Zhu‡
, Yi Chang] and Philip S. Yu, 2017

* Solving a somewhat similar problem as E2Efold

* Cardinality constraints formulated as matrix inequality, see last equation in Section 3.3

* 2-step approach:

    1. initial score is assumed to be produced from a linear model (thus close-form solution)

    2. constrained optimization between y and y_hat, solved by greedy algorithm,
    by selecting edge with largest score at each time and decide whether to add it
    (not a principled as E2Efold)

### ﻿Evaluating the Ability of LSTMs to Learn Context-Free Grammars

﻿Luzi Sennhauser, ﻿Robert C. Berwick, 2018

* Interesting paper with interesting experiments.
Conclusion: LSTM does not learn the underlying structure, its performance is from learning the sequential correlations.

* Evaluated LSTM performance and analyzed what's stored in intermediate states,
by training on bracket prediction task.

* bracket prediction task: predicting the next character in a string generated by Dyck language.
(﻿Every model which recognizes or generates well-formed Dyck words
with two types of brackets should be powerful enough to
handle any CFL when intersected with a relabeling)

* grammar has 4 production rules:

    ﻿- S -> S1 S | S1

    - S1 -> B | T

    - B -> [ S ] | { S }

    - T -> [ ] | { }

* To investigate the internal states of the model,
after training, encoder weights are fixed, and output labels are replaced by some features of input (up to that timestamp).
This reveals ﻿how accurately such feature about the input is preserved in the intermediate states.
Author investigated 2 features:

    - depth, a.k.a. nesting level, e.g. `﻿{[{}[[]` -> `1, 2, 3, 2, 3, 4, 3`

    - relevant/irrelavant input character, e.g. when previous inputs are ﻿`{[{}][`,
    `﻿[{}]` is irrelavant and the rest is relevant.

* To investigate generalization performance, the author selected a subset of training examples (in-sample),
e.g. strings with only even distances 2, 4, 6,.... and test on the rest (out-sample).

* Conclusion

    - memory demand grows exponentially with respect to the distance or depth of sentences that can be predicted

    - For generalization experiments, out-of-sample error rate is much higher than the in-sample one, see Fig 7

    - For intermediate state investigation, ﻿depth is only marginally conserved in the intermediate state, see Fig 8.
    But the models has learned to sort out relevant from irrelevant past inputs, see Fig 9.


### Deep learning of recursive structure: Grammar induction

Jason Eisner, ICLR 2013

talk video link not working: https://cs.jhu.edu/~jason/papers/

slides: https://cs.jhu.edu/~jason/papers/eisner.iclr13.pdf

Really interesting. Need to spend more time reading.

also see intro on grammar induction: https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-864-advanced-natural-language-processing-fall-2005/lecture-notes/lec11.pdf


### Compound Probabilistic Context-Free Grammars for Grammar Induction *



### ﻿Variational Inference for Adaptor Grammars


### ﻿SYNTAX-DIRECTED VARIATIONAL AUTOENCODER FOR STRUCTURED DATA





### Sum-Product Networks: A New Deep Architecture

Hoifung Poon and Pedro Domingos, 2011


CFG is SPN

to read

slides: https://vcla.stat.ucla.edu/sig11/sig11-domingos.pdf

also see https://homes.cs.washington.edu/~pedrod/papers/mlc13.pdf





### Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks

Kai Sheng Tai, Richard Socher, Christopher D. Manning, 2025


### Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations

Eliyahu Kiperwasser, Yoav Goldberg, 2016


### Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets

Armand Joulin, Tomas Mikolov, 2015


### Deep Learning with Dynamic Computation Graphs

Moshe Looks, Marcello Herreshoff, DeLesley Hutchins, Peter Norvig, 2017


## DNN + grammar?

## TODOs

