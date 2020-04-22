


Raw data [raw_data/PDB_dataset.zip](raw_data/PDB_dataset.zip) was downloaded from
paper SI: https://www.nature.com/articles/s41467-019-13395-9


`15` sequences were discarded due to the presence of ambiguous symbols.


## Dataset description


Used in "RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning"
as fine-tuning dataset.


```
The datasets for transfer learning were obtained by downloading high-resolution (<3.5 Å) RNAs from PDB on March 2, 20195.
Sequences with similarity of more than 80% among these sequences were removed with CD-HIT-EST37.
After removing sequence similarity, only 226 sequences remained.
These sequences were randomly split into 120, 30, and 76 RNAs for training (TR1),
validation (VL1), and independent test (TS1), respectively. Furthermore,
any sequence in TS1 having sequence similarity of more than 80% with TR0 was also removed,
which reduced TS1 to 69 RNAs. As CD-HIT-EST can only remove sequences with similarity more than 80%,
we employed BLAST-N40 to further remove potential sequence homologies with training data with a
large e-value cutoff of 10. This procedure further decreased TS1 from 69 to 67 RNAs.

To further benchmark RNA secondary-structure predictors, we employed 641 RNA structures solved by NMR.
Using CD-HIT-EST with 80% identity cutoff followed by BLAST-N with e-value cutoff of 10 against TR0, TR1, and TS1,
we obtained 39 NMR-solved structures as TS2.

The secondary structure of all the PDB sets was derived from their respective structures by using DSSR58 software.
For NMR- solved structures, model 1 structure was used as it is considered as the most reliable structure among all.
The numbers of canonical, noncanonical, and pseudoknot base pairs, and base multiplets (triplets and quartets) for
all the sets are listed in Supplementary Table 7.
```