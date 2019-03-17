#!/bin/bash

mkdir -p data
mkdir -p report

python corss_validate.py bHLH
python corss_validate.py ETS
python corss_validate.py E2F
python corss_validate.py RUNX
