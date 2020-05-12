
Focus on BBBC021 dataset

## TODOs

Do a high-level review of all papers listed at the bottom of page: https://data.broadinstitute.org/bbbc/BBBC021

Each benchmark dataset: objective, dataset size, how were data generated, example image


interpretability/robustness: anyone training multi-task with segmentation etc.? (maybe there's no labels for segmentation)

self-supervised?

adversarial attack?

out-of-sample detection?

requires heavy pre-processing? what if image contains several phenotypes?

image rotation? shift? is it robust?

Google paper citing Novartis paper
Oren -> Novartis -> google

https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007348
self-supervised

# Google's paper (put title here)


## High content imaging dataset BBBC021

images

well/plate

input/target/objective

number of images

sources

## Challenges




---- to be removed ----

## Benchmark dataset

### ﻿Broad Bioimage Benchmark Collection

- Author used ﻿BBBC013, BBBC014, BBBC015, BBBC016, and BBBC021.

- microscopy images with some sort of 'ground truth' (expected result)

- ﻿BBBC013: U2OS cells cytoplasm–nucleus translocation, binary classification,
whether Forkhead (FKHR-EGFP) fusion protein accumulates in the nucleus.
One image per channel (Channel 1 = FKHR-GFP; Channel 2 = DNA). Image size is 640 x 640 pixels.
See details and example images:   https://data.broadinstitute.org/bbbc/BBBC013/


- ﻿BBBC014: U2OS cells cytoplasm–nucleus translocation, binary classification,
cytoplasm to nucleus translocation of the transcription factor NFκB in MCF7 (human breast adenocarcinoma cell line) and A549 (human alveolar basal epithelial) cells in response to TNFα concentration.
For each well there is one field with two images: a nuclear counterstain (DAPI) image and a signal stain (FITC) image. Image size is 1360 x 1024 pixels.
https://data.broadinstitute.org/bbbc/BBBC014


- BBBC015: The images are of a human osteosarcoma cell line (U2OS) co-expressing beta2 (b2AR) adrenergic receptor and arrestin-GFP protein molecules. The receptor was modified-type that generates "vesicle-type" spots upon ligand stimulation.
one image for green channel and one image for crimson channel. Image size is 1000 x 768 pixels.
https://data.broadinstitute.org/bbbc/BBBC015


- BBBC016: This image set is of a Transfluor assay where an orphan GPCR is stably integrated into the b-arrestin GFP expressing U2OS cell line. After one hour incubation with a compound the cells were fixed with (formaldehyde).
one image for green channel (GFP) and one image for blue channel (DNA). Image size is 512 x 512 pixels.
https://data.broadinstitute.org/bbbc/BBBC016



- BBBC021: Human MCF7 cells – compound-profiling experiment.
Phenotypic profiling attempts to summarize multiparametric, feature-based analysis of cellular phenotypes of each sample so that similarities between profiles reflect similarities between samples.
(sounds familiar, is this also what insitro was doing?)
Profiling is well established for biological readouts such as transcript expression and proteomics. Image-based profiling, however, is still an emerging technology.
(interesting, so people treat this in analogy to say RNAseq?)
This image set provides a basis for testing image-based profiling methods wrt. to their ability to predict the mechanisms of action of a compendium of drugs. The image set was collected using a typical set of morphological labels and uses a physiologically relevant p53-wildtype breast-cancer model system (MCF-7) and a mechanistically distinct set of targeted and cancer-relevant cytotoxic compounds that induces a broad range of gross and subtle phenotypes.

The images are of MCF-7 breast cancer cells treated for 24 h with a collection of 113 small molecules at eight concentrations. The cells were fixed, labeled for DNA, F-actin, and Β-tubulin, and imaged by fluorescent microscopy as described [Caie et al. Molecular Cancer Therapeutics, 2010].


There are 39,600 image files (13,200 fields of view imaged in three channels) in TIFF format.

Labelling: A subset of the compound-concentrations have been identified as clearly having one of 12 different primary mechanims of action.

See https://data.broadinstitute.org/bbbc/BBBC021
Google's 2017 paper using this dataset (top performance): https://www.biorxiv.org/content/10.1101/161422v1