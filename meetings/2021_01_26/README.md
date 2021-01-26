
## S2 inference & eval

Continued from last week.


### debugging pruning + s2 inference: eval_s2.ipynb

non standard stems not pruned? min loop size not pruned? -> nope,
pruning working, plot was wrong since we didn't generate the correct db_str, due to pseudoknot.
Temp fix: alternating use of 3 sets of brackets '()', '[]', '{}' when generating the string.
Didn't want to spend too much time making a better fix since db_str itself is a handicapped system.
A proper way tp visualize is to use the base pairing matrix as the graph structure.

```
def stem2db_str(df_stem, seq_len):
    bracket_pairs = cycle([('(', ')'), ('[', ']'), ('{', '}')])

    db_str = ['.'] * seq_len
    for _, row in df_stem.iterrows():
        bb_x = int(row['bb_x'])
        bb_y = int(row['bb_y'])
        siz = int(row['siz_x'])
        siz_y = int(row['siz_y'])
        assert siz == siz_y
        bracket_pair = next(bracket_pairs)  # py3
        for i in range(siz):
            db_str[bb_x+i] = bracket_pair[0]
            db_str[bb_y-i] = bracket_pair[1]
    return ''.join(db_str)
```

![plot/update_bracket_for_forna_display.png](plot/update_bracket_for_forna_display.png)

### Added plot using networkx

```
def display_ss_graph(df_stem, seq):
    G = nx.Graph()
    nodes = []
    for i, base in enumerate(seq):
        nodes.append((i, {"label": base}))
    G.add_nodes_from(nodes)
    # backbone
    for i in range(len(seq)-1):
        G.add_edge(i, i+1)
    # hydrogen bonds
    for _, row in df_stem.iterrows():
        bb_x = int(row['bb_x'])
        bb_y = int(row['bb_y'])
        siz = int(row['siz_x'])
        siz_y = int(row['siz_y'])
        assert siz == siz_y
        for i in range(siz):
            G.add_edge(bb_x+i,bb_y-i)
    return G


G = display_ss_graph(df_pred[df_pred['bb_type'] == 'stem'], seq)
nx.draw(G, with_labels=True)
```

![plot/forna_vs_networkx.png](plot/forna_vs_networkx.png)

WIP how to display base on node?

WIP does not display nicely for complex structure. Need tweaks?



### handle cases where some bb type is empty

Fixed, e.g. no iloop:

```
CGUAUCCGCCCGGAUUA
...((((....))))..
```


## Batch Mode

- Added batch mode

    - Data loader with length grouping and batch padding

    - masking in attention weight matrix, and all subsequent layer activation

    - masked BCE loss

- added unified batch workflow (train/validation) and device switch


<!--local debug, inside `s2_training`:-->

<!--```-->
<!--mkdir result/debug/-->
<!--python train_s2.py --in_file ../data/synthetic_s1_pred_1000_t0p1_k1_features.npz --config debug_config.yml --out_dir result/debug/-->
<!--```-->


### run on 5000 dataset

(workstation):

```
mkdir -p result/synthetic_s2_5000/
CUDA_VISIBLE_DEVICES=2 python train_s2.py --in_file ../../2021_01_12/data/synthetic_s2_5000_features.npz --config config.yml --out_dir result/synthetic_s2_5000/
```


![plot/s2_5000_train_progress.png](plot/s2_5000_train_progress.png)
(best model at epoch 34, plot produced by s2_training/training_progress.ipynb)


TODO upload model?

### run on full dataset

? (need pruning) (also note the params used by s1 inference, make sure to reflect when running inference <- shall we save it in the wrapper as a known version?)

(run from previous week's folder since upstream data is there: ~/work/psi-lab-sandbox/meetings/2021_01_12/)

```
(root)
python model_utils/prune_stage_1.py --in_file data/synthetic_s1_pred_full_t0p1_k10_c0p8.pkl.gz --out_file data/synthetic_s1_pred_full_t0p1_k10_c0p8_pruned.pkl.gz --min_pixel_pred 1 --min_prob 0.1 --min_hloop_size 2 --discard_ns_stem
(s2_training/)
python make_dataset.py --in_file ../data/synthetic_s1_pred_full_t0p1_k10_c0p8_pruned.pkl.gz  --out_file ../data/synthetic_s2_full_features.npz
```

(back to this folder)

```
mkdir -p result/synthetic_s2_full/
CUDA_VISIBLE_DEVICES=2 python train_s2.py --in_file ../../2021_01_12/data/synthetic_s2_full_features.npz --config config.yml --out_dir result/synthetic_s2_full/
```

## TODOs

try a few more params for S1 comparison plot: (1) t=0.02, k=1,c=0, (2) t=0.1,k=0,c=0.9, (3) t=0.1,k=0,c=0.5, ….etc.
generate another random test dataset (use new data format with top right corner)
try t=0.000001
try t=0.000001 and k=2


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

bpRNA?

Sources:

- S1 training data, synthetic sequences: `ZQi8RT`


Intermediate:


## TODOs

- latent variable model

- when do we predict 'no structure'?

- s1 inference: running on longer sequence, can create a wrapper of the existing interface:
seq -> short seq pairs -> dfs -> translate -> stitch -> prediction. Be careful with boundary effect.
bb across boundary > include all.

- s2 idea: stacked 2D map: seq + binary, one for each local structure (predicted by s1). self attn across 2d maps?

- s2 idea: GNN? 'meta' node connects local structure? predict on/off of meta node? still can't incoportate non-local structure

- S1 inference pipeline for super long sequences: break into chunks and stitch

- dataset: '../2020_11_24/data/rfam151_s1_pruned.pkl.gz'  'data/synthetic_s1_pruned.pkl.gz'

- inference pipeline: deal with cases where some types of bb are empty

- inference pipeline debug + improvement: n_proposal_norm > 1, implementation using queue, terminate condition

- s2 training: stems only? how to pass in background info like sequence? memory network? encoding?

- s2 training: add batch mode (debug to make sure it works), save model, set up inference utils so we can run the model

- s2 inference: greedy sampling with hard constraints (white & black list)

- s2 training dataset, for those example where s1 bb sensitivity < 100%, add in the ground truth bbs for contructing dataset for s2.
How to set features like median_prob and n_proposal_norm? Average in the same example?

- rfam151 (and other dataset): evaluate base pair sensitivity and specificity (allow off by 1?)

- evaluate sensitivity if we allow +/-1 shift/expand of each bb

- if above works and we have a NN for stage 2, we can feed in this extended set of bb proposals!

- attention -> output set?

- stage 1 prevent overfitting (note that theoretical upper bound is not 100% due to the way we constructed the predictive problem)

- upload best model to DC?

- evaluate rfam stage 2 predictions, majority are not identical, but are they close enough?

- investigate pseudoknot predictions, synthetic dataset (45886-32008)

- try running RNAfold and allow C-U and U-U (and other) base pairs, can we recover the lower FE structure that our model predicts?

- rfam151 dataset debug, is the ground truth bounding box correct? (make sure there’s no off-by-1 error)

- stage 1 model: iloop size = 0 on my side is bulge, make sure we have those cases!

- RNAfold performance on rfam151

- to debug: index 0 with length 117 and n_bbs 21 seems to be stuck during parsing.: python model_utils/run_stage_2.py --in_file data/rfam151_s1_bb_0p1.pkl.gz --out_file data/debug.pkl.gz --min_pixel_pred 3 --min_prob 0.5

- to debug: rfam151, RF00165_A, global structure contain invalid ones (implied iloop and hloop not included):
```
   bb_x  bb_y  siz_x  siz_y bb_type  n_proposal  prob_median  n_proposal_norm
0     1    17      2      2    stem           4     0.137343              1.0
1     4    45      8      8    stem          64     0.859667              1.0
2    27    57     10     10    stem         100     0.721043              1.0
.((.((((((((....)).........((((((((((.))))))))..)))))))))).... 14.362449399658637 100007.1
```

- stage 2, pick the first bb by sampling from all bb's (proportional to the 'likelihood' of the bb?),
then the next ones are picked by some attention based NN? black list & white list?

- extra constraints in stages 2? stem box needs to satisfy G-C, A-U, G-U base pairing (discard those that are not),
min hloop size?

- table documenting all DC IDs (datasets, models, etc.)


- Heuristics: More structure is better -> if global struct A is subset of B, discard A

- Pseudo knot?

- RNA-RNA interaction? Run stage 1 model three times, A-A, B-B & A-B, 2nd stage will have different constraints

- Long sequence?

- Greedy approach of assembly? Start with high prob bounding boxes, terminate after explored say 100 global structures?

- size > 10






old dataset in top left corner format, convert everything to top right?

