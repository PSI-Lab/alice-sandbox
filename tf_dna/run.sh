#!/bin/bash

mkdir -p data
mkdir -p report

python cross_validate.py bHLH
python cross_validate.py ETS
python cross_validate.py E2F
python cross_validate.py RUNX
