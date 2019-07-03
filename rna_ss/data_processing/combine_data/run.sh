#!/usr/bin/env bash

python make_data.py --input_data raw_data/wan2014/GM12878_renatured.tab.gz --output_data_reactivity data/reactivity_human_﻿lymphoblastoid_family_child.gtrack --output_data_coverage data/coverage_human_﻿lymphoblastoid_family_child.gtrack --output_intervals data/intervals_human_﻿lymphoblastoid_family_child.txt --data_type pars --config config.yml
