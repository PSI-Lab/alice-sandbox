# 2020-02-07


- evaluate my model on their test set (how to generate one structure?)

- conv filter visualization (before inner product)

- 0.25/0.75 encoding

- add non-recurrent target (predict single structure)

- random recurrent directions


- other ways to predict single structure from distribution?

- eval dataset from that paper

## yeast data

Explore data used in paper: "Using deep learning to model the hierarchical structure and function of a cell".

- Online tool made by the authors: http://d-cell.ucsd.edu/

- GitHub: https://github.com/idekerlab/DCell

- Ontology: http://geneontology.org/docs/download-ontology/

- Genetic interaction and growth: https://thecellmap.org/costanzo2016/

    - From Bonne lab, ~23 million double mutants

    - 3 types of interactions: nonessential x nonessential, essential x nonessential, essential x essential

    - File download & schema description at: https://thecellmap.org/costanzo2016/

    - Each yeast strain correspond to one gene and one allele, how are alleles different from each other?

- File: http://thecellmap.org/costanzo2016/data_files/Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip

    - `strain_ids_and_single_mutant_fitness.xlsx` has information on 10859 strains,
    with their gene and allele ID, fitness at different temperature, and fitness std at 30C.

    ```
                 Strain ID Systematic gene name Allele/Gene name  Single mutant fitness (26°)  Single mutant fitness (26°) stddev  Single mutant fitness (30°)  Single mutant fitness (30°) stddev
    0  YAL001C_damp174              YAL001C        tfc3_damp                       0.9395                              0.0988                       0.9395                              0.0988
    1   YAL001C_tsa508              YAL001C       tfc3-g349e                       0.8658                              0.1789                       0.8285                              0.1964
    ```

    - `SGA_ExE.txt` has 818570 interactions, with strain and allele ID for both the query and array,
    as well as the double mutatnt fitness and std.

    ```
          Query Strain ID Query allele name  Array Strain ID Array allele name Arraytype/Temp  Genetic interaction score (ε)   P-value  Query single mutant fitness (SMF)  Array SMF  Double mutant fitness  Double mutant fitness standard deviation
    0  YAL001C_tsq508        tfc3-g349e   YBL023C_tsa111            mcm2-1          TSA30                        -0.0348  0.005042                             0.8285     0.9254                 0.7319                                    0.0102
    1  YAL001C_tsq508        tfc3-g349e  YBL026W_tsa1065         lsm2-5001          TSA30                        -0.3529  0.000004                             0.8285     0.9408                 0.4266                                    0.0790
    ```

    - `SGA_ExN_NxE.txt` has 5796145 interactions. Format is same as above.

    - `SGA_NxN.txt` has 12698939 interactions. Format is same as above.

    - `SGA_DAmP.txt`  has 1391958 interactions. Format is same as above. (Genetic interactions involving DAmP alleles of essential genes, what's this?)

- File http://thecellmap.org/costanzo2016/data_files/Raw%20genetic%20interaction%20datasets:%20Matrix%20format.zip

    - matrix format for the 3 types of interaction

    - .cdt, .atr and .gtr seem to be clustering output


- Yeast Interactome project: http://interactome.dfci.harvard.edu/S_cerevisiae/

    - `http://interactome.dfci.harvard.edu/S_cerevisiae/download/Y2H_union.txt` pairwise gene IDs





- GO: TODO

- Yeast gene network: where to find it? can we use it to initialize the structure?

- how to combine different modalities: PPI, etc.

- to read: https://www.nature.com/articles/srep04273

- to read: https://www.pnas.org/content/114/46/12166

- model 'trans factors', environment

- auxiliary training target

- Can we do better than what's in the paper "Using deep learning to model the hierarchical structure and function of a cell"?



## Multi-view network

- "Multi-GCN: Graph Convolutional Networks for Multi-View Networks, with Applications to Global Poverty"

- networks in which nodes can be related in multiple ways


