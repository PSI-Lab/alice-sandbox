

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117840


```bash
python make_data.py
gzip data/hek293_icshape_ch_vivo.csv

python make_data_mouse.py raw_data/GSM3310478_mes_ch_vivo.out.txt.gz data/mouse_esc_v65_icshape_ch_vivo.csv
gzip data/mouse_esc_v65_icshape_ch_vivo.csv

python make_data_mouse.py raw_data/GSM3310481_mes_np_vivo.out.txt.gz data/mouse_esc_v65_icshape_np_vivo.csv
gzip data/mouse_esc_v65_icshape_np_vivo.csv


```


