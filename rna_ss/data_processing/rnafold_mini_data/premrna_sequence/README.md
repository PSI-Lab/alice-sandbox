

```bash
mkdir -p tmp_output
mkdir -p data
mkdir -p cmd
python make_sequences.py --minlen 10 --maxlen 200 --parts 10 --out tmp_output/data
```

```bash
python make_cmd.py 10 cmd/cmd.txt
```

``bash
parallel --jobs 2 < cmd/cmd.txt
```

```bash
python combine_data.py data/premrna_seqs_10_200.pkl.gz tmp_output
```

