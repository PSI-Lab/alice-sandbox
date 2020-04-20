## Workflow

1. Download and unzip the raw data, see [raw_data/](raw_data/).


2. Run the following to process data into our internal format:

```
python make_data.py
gzip data/bp_rna.pkl
```

3. Generated dataset was uploaded to DC: `RMw6xd`



Notes:

Was hoping to adapt https://github.com/cschu/biolib/blob/master/mdg_dt.py
but it only works with non-pseudoknot structure (supports only `(` and `)`).
Ended up writing my own util.


