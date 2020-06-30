
## ContraFold

https://github.com/csfoo/contrafold-se

(code and training/testing data)

Need to fix my c++ compiler.


## Literature review

Continued from last week: [../2020_06_16/README.md](../2020_06_16/README.md)

### ﻿Recurrent Neural Network Grammars

﻿Chris Dyer♠ Adhiguna Kuncoro♠ Miguel Ballesteros♦♠ Noah A. Smith, 2016

code:
c++: https://github.com/clab/rnng
pytorch: https://github.com/kmkurn/pytorch-rnng

podcast: https://soundcloud.com/nlp-highlights/04-recurrent-neural-network-grammars-with-chris-dyer


- explicitly models nested structure.

- reminiscent of probabilistic context-free grammar generation, but decisions are parameterized using RNNs
that condition on the entire syntactic derivation history, greatly relaxing context-free independence assumptions.

- parsing: sequence of words -> parse tree. A stack and a buffer.
Begins with empty stack and all words in buffer.
Three types of operations: NT(X), SHIFT and REDUCE. See Fig 2.

- generation. see Fig 4.

- ﻿sequence model defined over generator transitions, ﻿parameterized using a continuous space embedding
 of the algorithm state at each time step.
 Algorithm state represented by embedding of the three data structures:
 output buffer, stack, and history of actions. These are of variable length so use RNN to encode.



### ﻿Unsupervised Recurrent Neural Network Grammars

﻿Yoon Kim† Alexander M. Rush† Lei Yu3
Adhiguna Kuncoro‡,3 Chris Dyer3 G´abor Melis, 2019

(follow up work from above?)

code: https://github.com/harvardnlp/urnng



### Compound Probabilistic Context-Free Grammars for Grammar Induction *


### Gradient Estimation with Stochastic Softmax Tricks

The Gumbel-Max trick is the basis of many relaxed gradient estimators. These estimators are easy to implement and low variance, but the goal of scaling them comprehensively to large combinatorial distributions is still outstanding. Working within the perturbation model framework, we introduce stochastic softmax tricks, which generalize the Gumbel-Softmax trick to combinatorial spaces. Our framework is a unified perspective on existing relaxed estimators for perturbation models, and it contains many novel relaxations. We design structured relaxations for subset selection, spanning trees, arborescences, and others. When compared to less structured baselines, we find that stochastic softmax tricks can be used to train latent variable models that perform better and discover more latent structure.


### OptNet: Differentiable Optimization as a Layer in Neural Networks

solve Quadratic Programming

constrained optimization

### SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver


slides: https://powei.tw/satnet_slide.pdf

poster: https://powei.tw/satnet_poster.pdf

video: (timestamp 43:00) https://www.facebook.com/icml.imls/videos/3253466301345987/


- jointly learning constraints and solutions, where constraints are represented as logical structures,
which are ﻿expressed by satisfiability problems.

- MAXSAT: ﻿maximize the number of clauses satisfied

- ﻿use differentiable SDP relaxations to capture relationships between discrete variables, fast solver by coordinate descent.

- backward pass computed analytically, no need to unroll the forward pass and store the Jacobians






### ﻿Variational Inference for Adaptor Grammars


### ﻿SYNTAX-DIRECTED VARIATIONAL AUTOENCODER FOR STRUCTURED DATA


### Sum-Product Networks: A New Deep Architecture

### Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks


### Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations

Eliyahu Kiperwasser, Yoav Goldberg, 2016

### Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets

Armand Joulin, Tomas Mikolov, 2015


### Deep Learning with Dynamic Computation Graphs

Moshe Looks, Marcello Herreshoff, DeLesley Hutchins, Peter Norvig, 2017


### Deep learning of recursive structure: Grammar induction

Jason Eisner, ICLR 2013

talk video link not working: https://cs.jhu.edu/~jason/papers/

slides: https://cs.jhu.edu/~jason/papers/eisner.iclr13.pdf

Really interesting. Need to spend more time reading.

also see intro on grammar induction: https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-864-advanced-natural-language-processing-fall-2005/lecture-notes/lec11.pdf



### Adventures with RNA Graphs


## TODOs

reading contrafold + probing + try their code

read RNA as graph paper

learn how to process PDB data

find more details about deep learning grammar induction, his student? Henry Pao?

read paper, compound CFG

ask Alireza/Shreshth whether they are aware of any work that combines deep learning and grammar

