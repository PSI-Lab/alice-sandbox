# Check dataset overlap

Looking at both identical and highly similar sequences.


To reproduce, run:

Run a list of commands in parallel:

```bash
bash cmd.sh | parallel -j 8
```

Overlap IDs and scores will be generated for each pair of dataset, see result in [data/](data/).

Report summary:

```
python show_summary.py
```


Result (`n_similar\[n_identical\]`):


| overlap            | archive(3975) | bprna(13419) | pdb250(241) | rfam151(151) | rnastralign(37138) | sprocessed(5273) |
|--------------------|---------------|--------------|-------------|--------------|--------------------|------------------|
| archive(3975)      |               | 1873[1580]   | 35[22]      | 15[10]       | 13940[3796]        | 2029[1266]       |
| bprna(13419)       |               |              | 19[2]       | 120[60]      | 2000[1310]         | 1066[636]        |
| pdb250(241)        |               |              |             | 1[0]         | 55[22]             | 92[32]           |
| rfam151(151)       |               |              |             |              | 14[8]              | 6[2]             |
| rnastralign(37138) |               |              |             |              |                    | 1438[574]        |
| sprocessed(5273)   |               |              |             |              |                    |                  |



