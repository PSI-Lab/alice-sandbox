http://www.rnasoft.ca/CG/


`Processed.txt` cannot be downloaded =(



process_dataset_s_test.py


## Dataset description

### rfam_151

Raw data was downloaded from http://www.rnasoft.ca/CG/.
It was used in "CONTRAfold: RNA secondary structure prediction without
physics-based models" as training and validation data.
It contains one sequence for each of the 151 noncoding RNA families.



```
To assess the suitability of CLLMs as models for RNA secondary
structure, we performed a series of cross-validation experiments
using known consensus secondary structures of noncoding
RNA families taken from the Rfam database [5,6]. Specifically,
version 7.0 of Rfam contains seed multiple alignments for 503
noncoding RNA families, and consensus secondary structures
for each alignment either taken from a previously published
study in the literature or predicted using automated covariancebased methods.

To establish ‘‘gold-standard’’ data for training and testing, we
first removed all seed alignments with only predicted secondary
structures, retaining the 151 families with secondary structures
from the literature. For each of these families, we then projected
the consensus family structure to every sequence in the alignment,
and retained the sequence/structure pair with the lowest combined
proportion of missing nucleotides and non-{au, cg, gu} base pairs.
The end result was a set of 151 independent examples, each taken
from a different RNA family.
```


### s_processed


Raw data was downloaded from http://www.rnasoft.ca/CG/.
It was used in "Efficient parameter estimation for RNA secondary
structure prediction" as training data.
Note that the download link of S-Full is broken, here we only processed S-processed.
This dataset was compiled from other databases by the author.
Structures were processed by the author to remove non-canonical base pairs,
as well as any base pair that resulted in pseudoknot.
S-Processed contains sequence segments < 700nt.



```
The structural test set, S-Full, is a comprehensive RNA structural set that we assembled from databases of well-determined RNA
secondary structures. Table 4 shows the RNA families included in
this set, with their sizes and lengths, and references to the databases of provenance. Several preprocessing steps have been applied,
including removal of RNAs for archeae (which live in extreme
environments), unannotated loops or unknown nucleotides. Noncanonical base pairs and a minimal number of bases to resolve any
pseudoknots have been removed.

The training set, S-Processed, is similar to S-Full, but molecules longer than 700 nucleotides have been divided into shorter
sequences, so that the MFE structure prediction step is reasonably fast. Unannotated branches or branches containing unknown base
pairs have been truncated. For truncated structures, a restriction
string that restricts the cut ends to pair has been added; of these
structures, 66% have been included in S-Processed.
```

Above mentioned Table 4 lists the following RNA families:

    - tRNA

    - RNase P RNA

    - 5S rRNA

    - 16S rRNA

    - 23S rRNA

    - SRP RNA

    - Ribozymes

    - other

