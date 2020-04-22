#!/usr/bin/env bash

# each id correspond to one path
dataset_id=("rfam151" "sprocessed"  "pdb250"  "bprna"  "rnastralign"  "archive")
dataset_path=("../rna_cg/data/rfam.pkl"  "../rna_cg/data/s_processed.pkl"  "../pdb_250/data/pdb_250.pkl"  "../spot_rna/bp_rna/data/bp_rna.pkl.gz"  "../e2efold/data/rnastralign.pkl.gz"  "../e2efold/data/archiveII.pkl.gz")


for i in "${!dataset_id[@]}"
    do
        for j in "${!dataset_id[@]}"
            do
                if [ ${j} -gt ${i} ]
                    then
                        d1=${dataset_path[$i]}
                        d2=${dataset_path[$j]}
                        outf="data/${dataset_id[$i]}_${dataset_id[$j]}_overlap.csv.gz"
                        cmd="python compute_dataset_overlap.py --d1 $d1 --d2 $d2 --out_file $outf"
                        echo $cmd
                    fi
            done
    done
