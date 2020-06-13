

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




### ﻿Knudsen and Hein


## DNN + grammar?

## TODOs

