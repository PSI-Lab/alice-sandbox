#!/usr/bin/env bash

python make_data.py --input_data raw_data/K562_vitro.tab.gz --output_data_reactivity data/reactivity_k562_vitro.gtrack --output_data_coverage data/coverage_k562_vitro.gtrack --output_intervals data/intervals_k562_vitro.csv --data_type dms --config config.yml

python make_data.py --input_data raw_data/K562_vivo1_DMS_300.tab.gz --output_data_reactivity data/reactivity_k562_vivo_300.gtrack --output_data_coverage data/coverage_k562_vivo_300.gtrack --output_intervals data/intervals_k562_vivo_300.csv --data_type dms --config config.yml

python make_data.py --input_data raw_data/K562_vivo2_DMS_400.tab.gz --output_data_reactivity data/reactivity_k562_vivo_400.gtrack --output_data_coverage data/coverage_k562_vivo_400.gtrack --output_intervals data/intervals_k562_vivo_400.csv --data_type dms --config config.yml
