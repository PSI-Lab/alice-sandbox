#!/bin/bash

mkdir -p data
mkdir -p report
mkdir -p report/one_model

python corss_validate.py bHLH
python corss_validate.py ETS
python corss_validate.py E2F
python corss_validate.py RUNX

python corss_validate_one_model.py

