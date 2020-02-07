# 2020-02-07


- evaluate my model on their test set (how to generate one structure?)

- conv filter visualization (before inner product)

- 0.25/0.75 encoding

- random recurrent directions


- other ways to predict single structure from distribution?


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

- In recent years, many different algorithms have been proposed for learning on multi-view graphs.
These algorithms can be broadly divided into three main categories: 1) co-training algorithms,
2) learning with multiple kernels, and 3) subspace learning (See Xu, Tao, and Xu (2013) for a survey).
Recent work by Dong et al. (2014) show that subspace approaches — which
find a latent subspace shared by multiple views — perform
well relative to co-training and kernelized approaches on a
range of tasks.


- need to further read the paper


## Add non-recurrent target (predict single structure)

https://github.com/PSI-Lab/alice-sandbox/tree/2d2b1f0d250f208f89e57f39ae1e89fab067fb48/rna_ss/model/rnafold_mini_data_2d_ar_4

Running

```
CUDA_VISIBLE_DEVICES=0  python train.py  --config config.yml --data /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/rnafold_mini_data/make_data_var_len/data/rand_seqs_var_len_sample_mfe_10_200_1000000.pkl.gz --output model/2020_02_02.1
```


Generating prediction using one of the trained model (training still running):

https://github.com/PSI-Lab/alice-sandbox/tree/4b09678328ba9bb28f01d753a381b97081eda103/rna_ss/eval/rnafold_mini_data_2d_ar_4

```
CUDA_VISIBLE_DEVICES=1 python make_prediction.py --model /home/alice/work/psi-lab-sandbox/rna_ss/model/rnafold_mini_data_2d_ar_4/run_2020_02_02_17_31_09/checkpoint.016.hdf5 --dataset /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/spot_rna/fine_tuning/data/spot_rna_fine_tuning.pkl --output result/debug.2020_02_06_1.pkl
```

Evaluate the above prediction by applying different threshold and computing metrics (above result file not checked in to Git, can be reproduced):

```
0.002
        f_measure  sensitivity         ppv
count  251.000000   255.000000  254.000000
mean     0.584506     0.516184    0.681916
std      0.199856     0.210456    0.234904
min      0.018868     0.000000    0.000000
25%      0.441919     0.370669    0.520682
50%      0.590164     0.516129    0.705882
75%      0.741799     0.651923    0.872917
max      1.000000     1.000000    1.000000
```

See https://github.com/PSI-Lab/alice-sandbox/blob/52c51e5443da849582751dd07438b580624210df/rna_ss/eval/rnafold_mini_data_2d_ar_4/result_analysis_2020_02_06.ipynb

Also observed that predictions are being pushed towards 0, might need to reweight labels during training.

## Check their test set TS1

Above section focused on the entire fine tuning set, here we focus on TS1 (reported in the paper).

Re-generate dataset with TR1/TS1/etc. labels:

https://github.com/PSI-Lab/alice-sandbox/tree/cc1ad30ee3b19fa948db49b7daed75d5b9db3c43/rna_ss/data_processing/spot_rna/fine_tuning

```
python make_data.py
```

Using this to make prediction:

```
CUDA_VISIBLE_DEVICES=1 python make_prediction.py --model /home/alice/work/psi-lab-sandbox/rna_ss/model/rnafold_mini_data_2d_ar_4/run_2020_02_02_17_31_09/checkpoint.016.hdf5 --dataset /home/alice/work/psi-lab-sandbox/rna_ss/data_processing/spot_rna/fine_tuning/data/spot_rna_fine_tuning.pkl --output result/debug.2020_02_06_2.pkl
```


Focus on their test set TS1:

```
0.002
       f_measure  sensitivity        ppv
count  66.000000    67.000000  67.000000
mean    0.569079     0.502614   0.659169
std     0.208150     0.217609   0.234258
min     0.126582     0.000000   0.000000
25%     0.412903     0.351247   0.536316
50%     0.565385     0.472222   0.687500
75%     0.726010     0.666667   0.819293
max     1.000000     1.000000   1.000000
```

Similar to the whole set. See https://github.com/PSI-Lab/alice-sandbox/blob/9b906b7ddd155082a03f75c330e1a5db4c2dd1d5/rna_ss/eval/rnafold_mini_data_2d_ar_4/result_analysis_2020_02_06.ipynb




TODO generate prediction using SPOT-RNA model (be careful, above is their fine-tuning set):

/Users/alicegao/work/psi-lab-sandbox/rna_ss/meetings/2020_01_31/SPOT-RNA

TODO analyze the result at /Users/alicegao/work/psi-lab-sandbox/rna_ss/meetings/2020_01_31/SPOT-RNA/outputs (clean up and re-run prediction)



TODO fine-tune on the above set


## Eval dataset

- eval dataset used in "﻿RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning"

- fine-tuning: https://www.dropbox.com/s/srcs1wx91qyyszp/PDB_dataset.zip
Processed dataset: https://github.com/PSI-Lab/alice-sandbox/tree/7c236f1e287193454f7d361a9878895d1632dc64/rna_ss/data_processing/spot_rna/fine_tuning

- couldn't find the test dataset, might need to contact authors


## TODOs

- generate prediction using SPOT-RNA model on TS1, evaluate and compare on each sequence (in progress)

- generate prediction using SPOT-RNA model on other dataset (rfam?)

- train/fine tune on their training dataset

- ask author for test dataset

- 0.25/0.75 encoding

- conv filter visualization (before inner product)

- other ways to predict single structure from distribution?

- random recurrent directions

- yeast data, possibility to collaborate? ask Brendan?

- reproduce yeast model, prototype GNN

- genotype -> phenotype. WGS -> RNAseq? prior work?

