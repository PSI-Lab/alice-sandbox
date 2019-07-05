# Meeting 2019-07-05

## Data Quality Check

* Made combined dataset from multiple studies, multiple cell lines: K562, HepG2, HEK293T, lymphoblast, fibroblast

* Good correlation between reps of same cell line

* Good correlation between vitro and vivo of same cell line

* Lower correlation between different cell line

* For more result, see [../../data_processing/combine_data/analysis.ipynb](../../data_processing/combine_data/analysis.ipynb)


## Model Training

* Multi-task with `11` sigmoid output, for different cell line and reps

* Specific cell line seems to be hard to train on

* See [../../model/multi_task/](../../model/multi_task/)

## Evaluation

* Cross-validation performance on K562 is lower than the previous model trained on K562 alone (expected)

* For high abundance transcripts, per-exon correlation & AUC higher than rnafold (transcript selection based on K562, need update)

* AUC on solved structure (mostly non-human) lower than rnafold

* See [../../eval/multi_task/eval.ipynb](../../eval/multi_task/eval.ipynb)

## TODOs

* Make sure the model is not just picking on some trivial signal, e.g. G/C

* Hyperparam search (model training takes much longer now)

* Process and train on mouse dataset

* Ribosnotch dataset?

* Train cell type specific model (e.g. reps + vivo + vitro), use correlation between rep to weight each transcript

* Re-process HEK293T dataset

* Re-run eval using different transcripts for cell lines

* Some studies also did targeted probing, can be used as validation data?

## Extra

TODOs:

intervals.txt also save transcript ID

### Data processing

TODO add SRA download instructions, combine all workflow


tmp notes

needed for all workflow:
BOWTIE_INDEX='/dg-shared/for_omar/data/SS_RNA_mapping/rf_indexes/refgene_rank_index/rank_1/index'
TX_FASTA='/dg-shared/for_omar/data/SS_RNA_mapping/rf_indexes/refgene_rank_index/rank_1/refgene_rank1.fa'
BED_FILE='/dg-shared/for_omar/data/SS_RNA_mapping/rf_indexes/refgene_rank_index/rank_1/refgene_rank1.bed'


TODO check suggested workflow in RNA framework doc

### Family trio (PARS) (wan2014)

Runs:

    Run     cell_line       sample_type     source_name
    SRR972706       GM12878 polyA RNA       polyA RNA S1 nuclease at 37°C
    SRR972707       GM12878 polyA RNA       polyA RNA Rnase V1 at 37°C
    SRR972708       GM12891 polyA RNA       polyA RNA S1 nuclease at 37°C
    SRR972709       GM12891 polyA RNA       polyA RNA Rnase V1 at 37°C
    SRR972710       GM12892 polyA RNA       polyA RNA S1 nuclease at 37°C
    SRR972711       GM12892 polyA RNA       polyA RNA Rnase V1 at 37°C
    SRR972712       GM12878 rRNA depleted   rRNA depleted RNA S1 nuclease at 37°C
    SRR972713       GM12878 rRNA depleted   rRNA depleted RNA Rnase V1 at 37°C
    SRR972714       GM12878 rRNA depleted   rRNA depleted RNA S1 nuclease at 37°C
    SRR972715       GM12878 rRNA depleted   rRNA depleted RNA Rnase V1 at 37°C
    SRR972716       GM12878 rRNA depleted   rRNA depleted RNA S1 nuclease at 37°C
    SRR972717       GM12878 rRNA depleted   rRNA depleted RNA Rnase V1 at 37°C

FASTQ_GZ_FILES='runs/SRX346863/SRR972714_1.fastq.gz runs/SRX346864/SRR972715_1.fastq.gz runs/SRX346865/SRR972716_1.fastq.gz  runs/SRX346866/SRR972717_1.fastq.gz runs/SRX346855/SRR972706_1.fastq.gz runs/SRX346856/SRR972707_1.fastq.gz runs/SRX346857/SRR972708_1.fastq.gz runs/SRX346858/SRR972709_1.fastq.gz runs/SRX346859/SRR972710_1.fastq.gz runs/SRX346860/SRR972711_1.fastq.gz runs/SRX346861/SRR972712_1.fastq.gz runs/SRX346862/SRR972713_1.fastq.gz'

TODO: (above) only using one of the pair?

rf-map -bc 32000 -o $OUTPUT_MAP -p 4 -wt 12 -b2 -b5 3 -b3 51 -kl -bnr --bowtie-N 1 -bi $BOWTIE_INDEX $FASTQ_GZ_FILES

    * -bc: Maximum MB of RAM for best-first search frames
    * p & wt: process & threads
    * -b2: Uses Bowtie v2 for reads mapping
    * -b5: Number of bases to trim from 5'-end of reads
    * -b3: Number of bases to trim from 3'-end of reads
    * -kl: Disables logs folder deletion
    * -bnr: Maps only to transcript's sense strand
    * -bowtie-n: Use Bowtie mapper in -n mode (0-3, Default: 2)
       Note: in -n mode, Bowtie admits no more than -bn mismatches in the seed

rf-count -o $OUTPUT_COUNT -p 4 -wt 12 -t5 3 -r -nm -f $TX_FASTA $OUTPUT_MAP/*.bam

    * p & wt: process & threads
    * -t5: Number of bases to trim from 5'-end of reads
    * -r: In case SAM/BAM files are passed, assumes that they are already sorted lexicographically by transcript ID, and numerically by position
    * -nm: Disables counting of total mapped reads


Data to use:

/dg-shared/for_omar/data/SS_RNA_mapping/studies/wan2014/out/rf_clean_1/

    -rwxrwxrwx 1 omar users 357M Aug  3  2018 GM12878_renatured.tab.gz*
    -rwxrwxrwx 1 omar users 338M Aug  3  2018 GM12891_renatured.tab.gz*
    -rwxrwxrwx 1 omar users 354M Aug  3  2018 GM12892_renatured.tab.gz*


### HepG2 (PARS-seq) (solomon2017)

Runs:

    Run     source_name
    SRR5719996      S1_control_PARSseq
    SRR5719997      S1_KD_PARSseq
    SRR5719998      S1_control_PARSseq
    SRR5719999      S1_KD_PARSseq
    SRR5720000      V1_control_PARSseq
    SRR5720001      V1_KD_PARSseq
    SRR5720002      V1_control_PARSseq
    SRR5720003      V1_KD_PARSseq

(KD's are for investigating TODO, we're only using controls here <- WT cell line)

SRR5719996: TODO
SRR5719998: TODO
SRR5720000: TODO
SRR5720002: TODO

FASTQ_GZ_FILES='runs/SRX2937075/SRR5719996.fastq.gz runs/SRX2937078/SRR5719998.fastq.gz runs/SRX2937080/SRR5720000.fastq.gz runs/SRX2937082/SRR5720002.fastq.gz'

rf-map -bc 32000 -o out/rf_map_1 -p 4 -wt 12 -b2 -b5 3 -kl -bnr --bowtie-N 1 -bi $BOWTIE_INDEX $FASTQ_GZ_FILES

    * -bc: Maximum MB of RAM for best-first search frames
    * p & wt: process & threads
    * -b2: Uses Bowtie v2 for reads mapping
    * -b5: Number of bases to trim from 5'-end of reads
    * -kl: Disables logs folder deletion
    * -bnr: Maps only to transcript's sense strand
    * -bowtie-n: Use Bowtie mapper in -n mode (0-3, Default: 2)
       Note: in -n mode, Bowtie admits no more than -bn mismatches in the seed


rf-count -o out/rf_count_1 -p 4 -wt 12 -t5 3 -r -nm -f $TX_FASTA out/rf_map_1/*.bam

    * p & wt: process & threads
    * -t5: Number of bases to trim from 5'-end of reads
    * -r: In case SAM/BAM files are passed, assumes that they are already sorted lexicographically by transcript ID, and numerically by position
    * -nm: Disables counting of total mapped reads



(similar for both reps, take one for example)
mv $OUTPUT_COUNT/SRR5719996.rc $OUTPUT_COUNT/HepG2_S1_rep1.rc
mv $OUTPUT_COUNT/SRR5720000.rc $OUTPUT_COUNT/HepG2_V1_rep1.rc
rf-rctools view -t out/rf_count_1/HepG2_S1_rep1.rc | gzip > out/rf_count_1/HepG2_S1_rep1.tab.gz
rf-rctools view -t out/rf_count_1/HepG2_V1_rep1.rc | gzip > out/rf_count_1/HepG2_V1_rep1.tab.gz

rf-norm -ow -p 48 -o out/rf_norm_1/HepG2_rep1/ -u out/rf_count_1/HepG2_V1_rep1.rc -t out/rf_count_1/HepG2_S1_rep1.rc -i out/rf_count_1/index.rci

    * -ow: Overwrites the output directory if already exists
    * -p: threads
    * -u: Path to the RC file for the non-treated/denatured (or RNase V1) sample
    * -t: Path to the RC file for the treated (or Nuclease S1) sample


Rscript ../wan2014/clean_wan2014.r out/rf_count_1/HepG2_S1_rep1.tab.gz out/rf_count_1/HepG2_V1_rep1.rc out/rf_norm_1/HepG2_rep1/ $BED_FILE out/rf_clean_1/HepG2_rep1.tab.gz out/rf_clean_1/HepG2_rep1_rm_nocoverage.tab.gz

Data to use:

/dg-shared/for_omar/data/SS_RNA_mapping/studies/solomon2017/out/rf_clean_1/

    -rwxrwxrwx 1 omar users 331M Sep  5  2018 HepG2_rep1.tab.gz*
    -rwxrwxrwx 1 omar users 331M Sep  5  2018 HepG2_rep2.tab.gz*

### HEK293 (DMS-MaPseq) (zubradt2016)

SRR3929619: small number of reads?
SRR3929620: actual data?

FASTQ_GZ_FILES='runs/SRX1959208/SRR3929619_trimmed.fq.gz runs/SRX1959208/SRR3929620_trimmed.fq.gz'

rf-map -bk 3 -bv 5 -b5 3 -cp -bs -b2 -kl -p 2 -wt 12 -bi $BOWTIE_INDEX -o out/rf_map_1 $FASTQ_GZ_FILES

    * -bk: Reports up to this number of mapping positions for reads
    * -bv: Use Bowtie mapper in -v mode (0-3, Default: disabled)
           Note: in -v mode, Bowtie admits no more than -bv mismatches in the entire read (Phred quality ignored)
    * -b5: Number of bases to trim from 5'-end of reads
    * -cp: Assumes that reads have been already clipped
    * -bs: Enables local alignment mode
    * -b2: Uses Bowtie v2 for reads mapping
    * -kl: Disables logs folder deletion

TODO the above is quite different from HelpG2 workflow, re-run?

rf-count -o out/rf_count_1 -p 4 -wt 12 -r -nm -f $TX_FASTA out/rf_map_1/*.bam

TODO norm?



TODO how to use the normalized data?

(added by me)
rf-rctools view -t out/rf_count_1/SRR3929620_trimmed.rc  | gzip > out/rf_count_1/SRR3929620_trimmed.tab.gz

Rscript clean_rouskin2014.r out/rf_count_1/SRR3929620_trimmed.tab.gz  out/rf_norm_1/ $BED_FILE out/rf_clean_1/hek293_2.tab.gz out/rf_clean_1/hek293_2_cov.tab.gz

new:

rf-count -o out/rf_count_new -p 4 -wt 12 -r -nm -f /dg-shared/for_omar/data/SS_RNA_mapping/rf_indexes/refgene_rank_index/rank_1/refgene_rank1.fa out/rf_map_1/SRR3929620_trimmed.bam

rf-rctools view -t out/rf_count_new/SRR3929620_trimmed.rc  | gzip > out/rf_count_new/SRR3929620_trimmed.tab.gz

rf-norm -t out/rf_count_new/SRR3929620_trimmed.rc -i out/rf_count_new/index.rci -sm 4 -nm 2 -rb AC -o out/rf_norm_new/
(only works in base env...)

Rscript clean_rouskin2014.r out/rf_count_new/SRR3929620_trimmed.tab.gz out/rf_norm_new/ /dg-shared/for_omar/data/SS_RNA_mapping/rf_indexes/refgene_rank_index/rank_1/refgene_rank1.bed out/rf_clean_new/hek293_2.tab.gz out/rf_clean_new/hek293_2_cov.tab.gz

Data to use:

/home/alice/work/psi-lab-sandbox/rna_ss/data_processing/zubradt2016/out/rf_clean_new/

    -rwxrwxrwx 1 alice users 266475360 Jul  3 10:18 hek293_2.tab.gz*


### K562 (DMS) (rouskin2014)

Runs (used ones only):

    Run     source_name
    SRR1057939      Human bone marrow, chronic myelogenous leukemia
    SRR1057940      Human bone marrow, chronic myelogenous leukemia
    SRR1057941      Human bone marrow, chronic myelogenous leukemia
    SRR1057942      Human bone marrow, chronic myelogenous leukemia
    SRR1057943      Human bone marrow, chronic myelogenous leukemia
    SRR1057944      Human bone marrow, chronic myelogenous leukemia
    SRR1057945      Human bone marrow, chronic myelogenous leukemia
    SRR1057946      Human bone marrow, chronic myelogenous leukemia
    SRR1058055      Human fibroblasts
    SRR1058056      Human fibroblasts
    SRR1058057      Human fibroblasts


FASTQ_GZ_FILES='runs/SRX398501/SRR1057939.fastq.gz runs/SRX398501/SRR1057940.fastq.gz runs/SRX398502/SRR1057941.fastq.gz runs/SRX398604/SRR1058055.fastq.gz runs/SRX398503/SRR1057942.fastq.gz runs/SRX398503/SRR1057943.fastq.gz runs/SRX398503/SRR1057944.fastq.gz runs/SRX398504/SRR1057945.fastq.gz runs/SRX398504/SRR1057946.fastq.gz runs/SRX398605/SRR1058056.fastq.gz runs/SRX398606/SRR1058057.fastq.gz'

rf-map -bc 32000 -o $OUTPUT_MAP -p 4 -wt 12 -b2 -b3 3 -kl -bnr --bowtie-N 1 -bi $BOWTIE_INDEX $FASTQ_GZ_FILES

    * -b2: Uses Bowtie v2 for reads mapping
    * -b3: Number of bases to trim from 3'-end of reads
    * -kl: Disables logs folder deletion
    * -bnr: Maps only to transcript's sense strand
    * -bowtie-n: Use Bowtie mapper in -n mode (0-3, Default: 2)
       Note: in -n mode, Bowtie admits no more than -bn mismatches in the seed


rf-count -o $OUTPUT_COUNT -p 4 -wt 12 -r -nm -f $TX_FASTA $OUTPUT_MAP/*.bam

merge:
rf-rctools merge $OUTPUT_COUNT/SRR1057939.rc $OUTPUT_COUNT/SRR1057940.rc -ow -i $OUTPUT_COUNT/index.rci -o $OUTPUT_COUNT/K562_vivo1_DMS_300.rc

rf-norm -p 48 -t $OUTPUT_COUNT/K562_vivo1_DMS_300.rc -i $OUTPUT_COUNT/index.rci -sm 2 -nm 2 -rb AC -o $OUTPUT_NORM/K562_vivo1_DMS_300/

    * -sm: Method for score calculation (1-4, Default: 1):
        1. Ding et al., 2014
        2. Rouskin et al., 2014
        3. Siegfried et al., 2014
        4. Zubradt et al., 2016
    * -nm: Method for signal normalization (1-3, Default: 1):
        1. 2-8% Normalization
        2. 90% Winsorizing
        3. Box-plot Normalization
    * -rb: Reactive bases to consider for signal normalization (Default: N [ACGT])
        Note: This parameter accepts any IUPAC code, or their combination (e.g. -rb M, or -rb AC). Any other base will be reported as NaN


Data to use:

/dg-shared/for_omar/data/SS_RNA_mapping/studies/rouskin2014/out/rf_clean_1/

    -rwxrwxrwx 1 omar users 413M Aug  3  2018 fibroblast_vitro.tab.gz*
    -rwxrwxrwx 1 omar users 363M Aug  3  2018 fibroblast_vivo.tab.gz*
    -rwxrwxrwx 1 omar users 392M Aug  3  2018 K562_vitro.tab.gz*
    -rwxrwxrwx 1 omar users 439M Aug  3  2018 K562_vivo1_DMS_300.tab.gz*
    -rwxrwxrwx 1 omar users 385M Aug  3  2018 K562_vivo2_DMS_400.tab.gz*

### Lu 2016

This paper is RNA-RNA interaction



