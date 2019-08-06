

raw_data

```bash
cd raw_data
curl -O ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60034/suppl/GSE60034_v65polyA%2BicSHAPE_NAI-N3_vivo.bw
bigWigToWig GSE60034_v65polyA%2BicSHAPE_NAI-N3_vivo.bw GSE60034_v65polyA%2BicSHAPE_NAI-N3_vivo.wig
cd ../

python make_data.py
gzip data/mouse_esc_v65_vivo.csv
```
