## Dataset description

Used as pre-training data in "RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning".

```
We trained our models of ResNets and LSTM networks by building a nonredundant
 set of RNA sequences with annotated secondary structure from bpRNA34 at 80%
 sequence-identity cutoff, which is the lowest sequence-identity cutoff allowed
 by the program CD-HIT-EST37 and has been employed previously by many studies
 for the same purpose38,39. This dataset of 13,419 RNAs after excluding those >80%
 sequence identities was further randomly divided into 10,814 RNAs for training (TR0),
  1300 for validation (VL0), and 1,305 for an independent test (TS0).
```

More details from "bpRNA: large-scale automated annotation and analysis of RNA secondary structure":

```
The seven databases that comprise the bpRNA-1m metadatabase include
Comparative RNA Web (CRW) (26), tmRNA database (27), tRNAdb (28),
Signal Recognition Particle (SRP) database (29), RNase P database (30), tRNAdb
2009 database (31), and RCSB Protein Data Bank (PDB)
(32), and all families from RFAM 12.2 (33).
Moreover, to
reduce duplication for further analysis, we created a subset
called bpRNA-1m(90), where we removed sequences with
>90% sequence similarity when there is at least 70% alignment coverage (34). The bpRNA-1m database currently has
102 318 RNA structures and the bpRNA-1m(90) subset
consists of 28 370 structures. For comparison, the RNA
STRAND v2.0 database has 4666 structures, with fewer
than 2000 structures when similarly filtered.
```


## Workflow

1. Download and unzip the raw data, see [raw_data/](raw_data/).


2. Run the following to process data into our internal format:

```
python make_data.py
gzip data/bp_rna.pkl
```

3. Generated dataset was uploaded to DC: `RMw6xd`



Notes:

Was hoping to adapt https://github.com/cschu/biolib/blob/master/mdg_dt.py
but it only works with non-pseudoknot structure (supports only `(` and `)`).
Ended up writing my own util.


