

## RNA SS baseline training

Tried training on rnastralign dataset (training dataset of E2Efold),
 see [rna_ss/](rna_ss/).

TODO:

- train model using non-redundant dataset (use same similarity cutoff as in the paper)

- comparison, use same metric (how to pick threshold for reporting metric?)

## Data processing


**archiveII** and **rnastralign**, used by E2Efold: done,
see [../../data_processing/e2efold/](../../data_processing/e2efold/).

**bp-RNA**, used by SPOT-RNA: in progress (need to parse dot-bracket format into binary matrix)

TODO:

- finish processing bp-RNA

- other dataset used by SPOT-RNA

## Data Analysis

TODO:

- dataset overlap

- sequence/structure similarity

- any possibility that existing methods overfit to certain RNA families?

- how is human mRNA different than other RNAs, e.g. rRNA?

- probing v.s. X-RAY/NMR (do we have data on huamn?)
