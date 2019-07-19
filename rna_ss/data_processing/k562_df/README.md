

python make_data.py --input_data raw_data/K562_vivo1_DMS_300.tab.gz --output_data data/k562_vivo1.csv  --data_type dms  --config config.yml

gzip data/k562_vivo1.csv


python make_data.py --input_data raw_data/K562_vivo2_DMS_400.tab.gz --output_data data/k562_vivo2.csv  --data_type dms  --config config.yml

gzip data/k562_vivo2.csv


python make_data.py --input_data raw_data/K562_vitro.tab.gz --output_data data/k562_vitro.csv  --data_type dms  --config config.yml

gzip data/k562_vitro.csv


python combine_data.py

gzip data/k562_combined.csv


TODO upload to DC

