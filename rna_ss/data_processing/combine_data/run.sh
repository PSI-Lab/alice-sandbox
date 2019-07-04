#!/usr/bin/env bash

# wan 2014

python make_data.py --input_data raw_data/wan2014/GM12878_renatured.tab.gz --output_data_reactivity data/reactivity_human_lymphoblastoid_family_child.gtrack --output_data_coverage data/coverage_human_lymphoblastoid_family_child.gtrack --output_intervals data/intervals_human_lymphoblastoid_family_child.txt --data_type pars --config config.yml

python make_data.py --input_data raw_data/wan2014/GM12891_renatured.tab.gz --output_data_reactivity data/reactivity_human_lymphoblastoid_family_father.gtrack --output_data_coverage data/coverage_human_lymphoblastoid_family_father.gtrack --output_intervals data/intervals_human_lymphoblastoid_family_father.txt --data_type pars --config config.yml

python make_data.py --input_data raw_data/wan2014/GM12892_renatured.tab.gz --output_data_reactivity data/reactivity_human_lymphoblastoid_family_mother.gtrack --output_data_coverage data/coverage_human_lymphoblastoid_family_mother.gtrack --output_intervals data/intervals_human_lymphoblastoid_family_mother.txt --data_type pars --config config.yml

# solomon 2017

python make_data.py --input_data raw_data/solomon2017/HepG2_rep1.tab.gz --output_data_reactivity data/reactivity_hepg2_rep1.gtrack --output_data_coverage data/coverage_hepg2_rep1.gtrack --output_intervals data/intervals_hepg2_rep1.txt --data_type pars --config config.yml

python make_data.py --input_data raw_data/solomon2017/HepG2_rep2.tab.gz --output_data_reactivity data/reactivity_hepg2_rep2.gtrack --output_data_coverage data/coverage_hepg2_rep2.gtrack --output_intervals data/intervals_hepg2_rep2.txt --data_type pars --config config.yml


# zubradt 2016

python make_data.py --input_data raw_data/zubradt2016/hek293_2.tab.gz --output_data_reactivity data/reactivity_hek293.gtrack --output_data_coverage data/coverage_hek293.gtrack --output_intervals data/intervals_hek293.txt --data_type dms --config config.yml


# rouskin 2014

python make_data.py --input_data raw_data/rouskin2014/fibroblast_vitro.tab.gz --output_data_reactivity data/reactivity_fibroblast_vitro.gtrack --output_data_coverage data/coverage_fibroblast_vitro.gtrack --output_intervals data/intervals_fibroblast_vitro.txt --data_type dms --config config.yml

python make_data.py --input_data raw_data/rouskin2014/fibroblast_vivo.tab.gz --output_data_reactivity data/reactivity_fibroblast_vivo.gtrack --output_data_coverage data/coverage_fibroblast_vivo.gtrack --output_intervals data/intervals_fibroblast_vivo.txt --data_type dms --config config.yml

python make_data.py --input_data raw_data/rouskin2014/K562_vitro.tab.gz --output_data_reactivity data/reactivity_k562_vitro.gtrack --output_data_coverage data/coverage_k562_vitro.gtrack --output_intervals data/intervals_k562_vitro.txt --data_type dms --config config.yml

python make_data.py --input_data raw_data/rouskin2014/K562_vivo1_DMS_300.tab.gz --output_data_reactivity data/reactivity_k562_vivo_300.gtrack --output_data_coverage data/coverage_k562_vivo_300.gtrack --output_intervals data/intervals_k562_vivo_300.txt --data_type dms --config config.yml

python make_data.py --input_data raw_data/rouskin2014/K562_vivo2_DMS_400.tab.gz --output_data_reactivity data/reactivity_k562_vivo_400.gtrack --output_data_coverage data/coverage_k562_vivo_400.gtrack --output_intervals data/intervals_k562_vivo_400.txt --data_type dms --config config.yml
